# locustfile.py
# -*- coding: utf-8 -*-
"""
Sock Shop load testing script (based on Kubernetes front-end Service)

Compatible with original script:
- Uses BASE_URL = http://front-end.sock-shop.svc.cluster.local:80
- Uses GET /login + Authorization: Basic ... for login
- Uses HTML page paths: /category.html, /detail.html?id=..., /basket.html, /customer-orders.html
- Shopping cart API: POST /cart, DELETE /cart/<productId>

New/Enhanced features:
- Fetches product list (JSON) via /catalogue for random product selection
- Multiple user types (Session-based behavior scripts), closer to real workload
- Introduces random behavior & probabilistic execution:
  - Some operations (checkout, add to cart, etc.) are executed based on random.random()
  - Random sleep between actions to simulate real user operation rhythm
- Realistic error simulation integrated into normal user flows:
  - ~5% chance of wrong password during login (with retry)
  - ~2% chance of malformed checkout payload
  - Errors occur organically as part of user journeys, not in isolation
"""

import os
import base64
import json
import random
import string  # For generating random passwords
import uuid
import time  # NEW: for random pauses
import logging
from typing import List, Dict

from locust import HttpUser, task, between

# ==============================
# Logging Configuration
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('locust_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==============================
# Global Configuration (compatible with original script)
# ==============================

BASE_URL = os.environ.get(
    # "BASE_URL", "http://front-end.sock-shop.svc.cluster.local:80"
    "BASE_URL", "http://sockshop.local:31728"
)

logger.info(f"Locust load testing initialized")
logger.info(f"Target BASE_URL: {BASE_URL}")
logger.info(f"Logging to file: locust_debug.log")

# Default account for Basic Auth (existing account in the system)
DEFAULT_USERNAME = os.environ.get("SS_USER", "Seabook")
DEFAULT_PASSWORD = os.environ.get("SS_PASS", "Seabook1111")

# If you previously fixed a specific product ID, keep it here:
DEFAULT_PRODUCT_ID = os.environ.get(
    "SS_PRODUCT_ID", "819e1fbf-8b7e-4f6d-811f-693534916a8b"
)

# Import shared user credentials
from user_credentials import BASE_USERS


def basic_auth_value(user: str, pwd: str) -> str:
    token = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


# Cache for product list from /catalogue (JSON)
PRODUCT_IDS: List[str] = []


def ensure_product_ids(user: "BaseSockShopUser") -> List[str]:
    """
    Ensure PRODUCT_IDS has been fetched from /catalogue.
    Calls API: GET /catalogue, returns JSON.
    """
    global PRODUCT_IDS
    if PRODUCT_IDS:
        logger.debug(f"Using cached product IDs: {len(PRODUCT_IDS)} products")
        return PRODUCT_IDS

    logger.info("Fetching product list from /catalogue")
    # /catalogue supports query parameters like ?size=100, but it's optional
    with user.client.get(
        "/catalogue",
        name="GET /catalogue (load products)",
        catch_response=True,
    ) as resp:
        if resp.status_code != 200:
            logger.error(f"Failed to fetch catalogue: status={resp.status_code}, response={resp.text[:200]}")
            resp.failure(f"status={resp.status_code}, text={resp.text[:200]}")
            return PRODUCT_IDS

        try:
            data = resp.json()
            logger.debug(f"Catalogue response received: {len(data)} items")
        except Exception as e:
            logger.error(f"Failed to parse catalogue JSON: {e}, response={resp.text[:200]}")
            resp.failure(f"parse json failed: {e}")
            return PRODUCT_IDS

        ids = []
        for item in data:
            pid = item.get("id")
            if pid:
                ids.append(pid)

        if not ids:
            logger.warning("No product IDs found in /catalogue response")
            resp.failure("no product ids found in /catalogue response")
        else:
            PRODUCT_IDS = ids
            logger.info(f"Successfully loaded {len(PRODUCT_IDS)} product IDs from catalogue")
            resp.success()

    return PRODUCT_IDS


def get_random_base_user() -> dict:
    """Randomly select one from the 20 pre-generated accounts"""
    return random.choice(BASE_USERS)


def generate_random_address() -> dict:
    """
    Generate random shipping address for testing.
    Returns a dictionary with address fields.
    """
    streets = [
        "Main Street", "Oak Avenue", "Maple Drive", "Cedar Lane", "Pine Road",
        "Elm Street", "Market Street", "High Street", "Park Avenue", "Broadway",
        "King Street", "Queen Street", "Church Road", "Station Road", "Mill Lane"
    ]
    cities = [
        "Bangkok", "London", "New York", "Tokyo", "Paris", "Berlin", "Sydney",
        "Toronto", "Singapore", "Amsterdam", "Stockholm", "Copenhagen", "Oslo",
        "Helsinki", "Dublin", "Edinburgh", "Manchester", "Melbourne", "Auckland"
    ]
    countries = [
        "Thailand", "United Kingdom", "United States", "Japan", "France",
        "Germany", "Australia", "Canada", "Singapore", "Netherlands",
        "Sweden", "Denmark", "Norway", "Finland", "Ireland", "New Zealand"
    ]

    # Generate random street number (1-999)
    street_number = str(random.randint(1, 999))

    # Generate random postcode (5-6 digits or alphanumeric)
    if random.random() < 0.5:
        postcode = f"{random.randint(10000, 99999)}"
    else:
        postcode = f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1, 9)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')} {random.randint(1, 9)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1, 9)}"

    return {
        "number": street_number,
        "street": random.choice(streets),
        "city": random.choice(cities),
        "postcode": postcode,
        "country": random.choice(countries),
    }


def generate_random_card() -> dict:
    """
    Generate random test credit card information.
    Uses valid test card number formats that pass Luhn check.
    Returns a dictionary with card fields.
    """
    # Use common test card numbers (these are standard test cards)
    test_card_prefixes = [
        "4111111111111111",  # Visa
        "4012888888881881",  # Visa
        "5555555555554444",  # Mastercard
        "5105105105105100",  # Mastercard
        "378282246310005",   # American Express
        "371449635398431",   # American Express
        "6011111111111117",  # Discover
        "6011000990139424",  # Discover
    ]

    # Generate random expiration date (future dates only)
    current_year = 26  # 2026
    exp_month = random.randint(1, 12)
    exp_year = current_year + random.randint(0, 8)  # Valid for next 0-8 years

    # Generate random CVV (3 or 4 digits)
    card_number = random.choice(test_card_prefixes)
    if card_number.startswith("3"):  # Amex uses 4-digit CVV
        ccv = str(random.randint(1000, 9999))
    else:
        ccv = str(random.randint(100, 999))

    return {
        "longNum": card_number,
        "expires": f"{exp_month:02d}/{exp_year}",
        "ccv": ccv,
    }


# ==============================
# Abstract Base Class: Common Operations
# ==============================

class BaseSockShopUser(HttpUser):
    """
    Base class shared by all user types:
    - Uses GET /login + Basic Auth for login
    - Provides common methods for page visits, cart operations, checkout, etc.
    - Introduces random pauses to simulate user thinking/scrolling/hesitation
    """
    # Locust built-in wait time between tasks (task level)
    wait_time = between(5, 10)
    host = BASE_URL
    abstract = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.account = None  # Account used by current virtual user (username/password)
        logger.info(f"Initialized {self.__class__.__name__} user")

    # ---------- Utility: Random pause between actions ----------

    def random_pause(self, min_s: float = 1.2, max_s: float = 3.5):
        """
        Sleep for a random duration between actions to simulate user operation rhythm:
        - Page browsing delay
        - Thinking / Hesitation
        - Form input, etc.
        """
        duration = random.uniform(min_s, max_s)
        time.sleep(duration)

    # ---------- Page Visits ----------

    def visit_home(self):
        self.client.get("/", name="GET /")
        self.random_pause()

    def visit_category_page(self):
        self.client.get("/category.html", name="GET /category.html")
        self.random_pause()

    def visit_basket_page(self):
        self.client.get("/basket.html", name="GET /basket.html")
        self.random_pause()

    def visit_orders_page(self):
        self.client.get("/customer-orders.html", name="GET /customer-orders.html")
        self.random_pause()

    def visit_detail_page_random(self):
        """Get product id from /catalogue, then visit /detail.html?id=..."""
        try:
            product_ids = ensure_product_ids(self)
            if not product_ids:
                # Fall back to default ID
                logger.warning("No product IDs available, using default product ID")
                pid = DEFAULT_PRODUCT_ID
            else:
                pid = random.choice(product_ids)

            logger.debug(f"Visiting product detail page: {pid}")
            self.client.get(
                f"/detail.html?id={pid}",
                name="GET /detail.html",
            )
            # Pause while viewing detail page
            self.random_pause()
            return pid  # Return the product id viewed this time
        except Exception as e:
            logger.error(f"Error in visit_detail_page_random: {e}", exc_info=True)
            self.random_pause()
            return DEFAULT_PRODUCT_ID

    # ---------- Login / Register ----------

    def login_with_basic(self, username: str, password: str, success_expected: bool = True):
        """
        Call GET /login with Basic Auth.
        """
        logger.debug(f"Attempting login for user: {username}, expect_success={success_expected}")
        headers = {"Authorization": basic_auth_value(username, password)}

        with self.client.get(
            "/login",
            headers=headers,
            name="GET /login",
            allow_redirects=True,
            catch_response=True,
        ) as resp:
            if success_expected:
                if resp.status_code == 200:
                    logger.info(f"Login successful for user: {username}")
                    resp.success()
                else:
                    logger.error(f"Login failed for user: {username}, status={resp.status_code}, response={resp.text[:200]}")
                    resp.failure(f"login failed: {resp.status_code} {resp.text[:200]}")
            else:
                # Intentionally create a failed login
                if resp.status_code == 200:
                    logger.warning(f"Login unexpectedly succeeded for invalid user: {username}")
                    resp.failure("login unexpectedly succeeded")
                else:
                    logger.debug(f"Login correctly failed for invalid user: {username}")
                    resp.success()
        self.random_pause()

    def login_default(self):
        """
        Login using default account (Seabook from original script).
        """
        self.login_with_basic(DEFAULT_USERNAME, DEFAULT_PASSWORD, success_expected=True)
        self.account = {"username": DEFAULT_USERNAME, "password": DEFAULT_PASSWORD}

    def login_existing_user(self):
        """
        Login using one of the 20 accounts (randomly select one if not already chosen).
        Small chance (~5%) of using wrong credentials to simulate user error.
        """
        if self.account is None:
            self.account = get_random_base_user()

        # 5% chance of entering wrong password (simulate user typo/mistake)
        if random.random() < 0.05:
            logger.debug(f"Simulating login error for user: {self.account['username']}")
            wrong_password = self.account["password"] + "_wrong"
            self.login_with_basic(
                self.account["username"], wrong_password, success_expected=False
            )
            # After failed login, try again with correct password (user realizes mistake)
            self.random_pause(0.5, 2.0)  # Longer pause as user realizes error
            logger.debug(f"Retrying login with correct credentials for user: {self.account['username']}")

        self.login_with_basic(
            self.account["username"], self.account["password"], success_expected=True
        )

    def register_new_user(self):
        """
        Call API /register to create a new user, store in self.account after success.
        Uses JSON format: username/password/email/firstName/lastName.
        After registration, also creates shipping address and payment card for checkout.
        """
        username = f"lt_{uuid.uuid4().hex[:8]}"

        # Generate random password (8-12 chars with mixed case, digits, special chars)
        password_length = random.randint(8, 12)
        password_chars = string.ascii_letters + string.digits + "!@#$%"
        password = ''.join(random.choice(password_chars) for _ in range(password_length))

        # Generate random first and last names
        first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
                       "Iris", "Jack", "Kate", "Leo", "Maya", "Noah", "Olivia", "Paul",
                       "Quinn", "Ryan", "Sophia", "Tom", "Uma", "Victor", "Wendy", "Xander",
                       "Yara", "Zoe", "Alex", "Blake", "Casey", "Drew"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
                      "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
                      "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
                      "Lee", "Thompson", "White", "Harris", "Clark", "Lewis", "Walker", "Hall"]

        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        email = f"{username}@example.com"

        logger.debug(f"Attempting to register new user: {username}")
        payload = {
            "username": username,
            "password": password,
            "email": email,
            "firstName": first_name,
            "lastName": last_name,
        }

        with self.client.post(
            "/register",
            name="POST /register",
            json=payload,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                self.account = {"username": username, "password": password}
                logger.info(f"Successfully registered new user: {username}")
                resp.success()
            else:
                logger.error(f"Registration failed for user: {username}, status={resp.status_code}, response={resp.text[:200]}")
                resp.failure(f"register failed: {resp.status_code} {resp.text[:200]}")
                return  # Don't create address/card if registration failed

        self.random_pause()

        # Create shipping address (required for checkout)
        self._create_address_for_user()

        # Create payment card (required for checkout)
        self._create_card_for_user()

    def _create_address_for_user(self):
        """Create shipping address for the current logged-in user"""
        address_payload = generate_random_address()
        logger.debug(f"Creating shipping address for new user: {address_payload['city']}, {address_payload['country']}")

        with self.client.post(
            "/addresses",
            name="POST /addresses",
            json=address_payload,
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 201):
                logger.info(f"Successfully created shipping address: {address_payload['street']}, {address_payload['city']}")
                resp.success()
            else:
                logger.error(f"Failed to create address: status={resp.status_code}, response={resp.text[:200]}")
                resp.failure(f"create address failed: {resp.status_code}")
        self.random_pause()

    def _create_card_for_user(self):
        """Create payment card for the current logged-in user"""
        card_payload = generate_random_card()
        logger.debug(f"Creating payment card for new user: {card_payload['longNum'][:4]}...{card_payload['longNum'][-4:]}")

        with self.client.post(
            "/cards",
            name="POST /cards",
            json=card_payload,
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 201):
                logger.info(f"Successfully created payment card ending in {card_payload['longNum'][-4:]}")
                resp.success()
            else:
                logger.error(f"Failed to create card: status={resp.status_code}, response={resp.text[:200]}")
                resp.failure(f"create card failed: {resp.status_code}")
        self.random_pause()

    # ---------- Shopping Cart / Checkout ----------

    def add_to_cart(self, product_id: str):
        """
        POST /cart, body: {"id": "<productId>"}, consistent with original script
        """
        logger.debug(f"Adding product to cart: {product_id}")
        add_body = {"id": product_id}
        with self.client.post(
            "/cart",
            name="POST /cart",
            data=json.dumps(add_body),
            headers={"Content-Type": "application/json"},
            catch_response=True,
        ) as resp:
            if resp.status_code not in (200, 201):
                logger.error(f"Add to cart failed: product_id={product_id}, status={resp.status_code}, response={resp.text[:200]}")
                resp.failure(
                    f"add to cart failed: {resp.status_code} {resp.text[:200]}"
                )
            else:
                logger.debug(f"Successfully added product to cart: {product_id}")
                resp.success()
        self.random_pause()

    def delete_from_cart(self, product_id: str):
        """
        DELETE /cart/<productId>, consistent with original script
        """
        logger.debug(f"Deleting product from cart: {product_id}")
        with self.client.delete(
            f"/cart/{product_id}",
            name="DELETE /cart/:id",
            catch_response=True,
        ) as resp:
            if resp.status_code not in (200, 202, 204):
                logger.error(f"Delete from cart failed: product_id={product_id}, status={resp.status_code}, response={resp.text[:200]}")
                resp.failure(
                    f"delete from cart failed: {resp.status_code} {resp.text[:200]}"
                )
            else:
                logger.debug(f"Successfully deleted product from cart: {product_id}")
                resp.success()
        self.random_pause()

    def checkout(self, expect_success: bool = True, execute_prob: float = 1.0):
        """
        Call backend checkout API: POST /orders.

        - execute_prob: Execution probability (0.0~1.0)
          For example, execute_prob=0.7 means only 70% chance to actually checkout,
          the remaining 30% simulates "abandoning checkout midway".

        This provides a "comprehensive" payload with random address and card info.
        Small chance (~2%) of submitting malformed data to simulate user/client errors.
        If you get 4xx/5xx responses, you can adjust according to your actual orders service requirements.
        """
        # Introduce probabilistic execution: some chance user abandons checkout
        if random.random() > execute_prob:
            # Don't execute checkout, simulate user closing page / abandoning payment
            logger.debug(f"Checkout abandoned (probability check failed, execute_prob={execute_prob})")
            return

        # 2% chance of submitting malformed payload (simulate client-side error, network corruption, etc.)
        if random.random() < 0.02 and expect_success:
            logger.debug("Simulating malformed checkout payload (client error)")
            bad_payload = {"incomplete": "data", "error": "simulation"}
            with self.client.post(
                "/orders",
                name="POST /orders",
                json=bad_payload,
                catch_response=True,
            ) as resp:
                # Expect this to fail
                if 200 <= resp.status_code < 300:
                    logger.warning(f"Malformed checkout unexpectedly succeeded: status={resp.status_code}")
                    resp.failure("malformed payload unexpectedly succeeded")
                else:
                    logger.debug(f"Malformed checkout correctly rejected: status={resp.status_code}")
                    resp.success()  # Success means it correctly rejected bad data
            self.random_pause()
            return

        # Generate random address and card for this checkout
        random_address = generate_random_address()
        random_card = generate_random_card()

        logger.debug(f"Attempting checkout with random info: {random_address['city']}, card ending {random_card['longNum'][-4:]}")
        payload = {
            "customer": {
                "firstName": "Load",
                "lastName": "Tester",
                "email": "load@test.com",
            },
            "address": random_address,
            "card": random_card,
        }

        # Log the complete payload for debugging
        logger.debug(f"Checkout payload being sent:\n{json.dumps(payload, indent=2)}")

        # Set explicit headers to avoid 406 Not Acceptable error
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        with self.client.post(
            "/orders",
            name="POST /orders",
            json=payload,
            headers=headers,
            catch_response=True,
        ) as resp:
            if expect_success:
                if 200 <= resp.status_code < 300:
                    logger.info(f"Checkout successful: status={resp.status_code}")
                    resp.success()
                else:
                    # Enhanced error logging for debugging
                    logger.error(f"Checkout failed: status={resp.status_code}")
                    logger.error(f"Response headers: {dict(resp.headers)}")
                    logger.error(f"Response body: {resp.text}")
                    logger.error(f"Request payload was:\n{json.dumps(payload, indent=2)}")

                    # Special diagnostic for 406 errors
                    if resp.status_code == 406:
                        logger.error("406 Not Acceptable - Server is rejecting the request")

                    resp.failure(
                        f"checkout failed: {resp.status_code} {resp.text[:200]}"
                    )
            else:
                if 200 <= resp.status_code < 300:
                    logger.warning(f"Checkout unexpectedly succeeded when failure was expected")
                    resp.failure("checkout unexpectedly succeeded")
                else:
                    logger.debug(f"Checkout correctly failed as expected: status={resp.status_code}")
                    resp.success()
        self.random_pause()


# ==============================
# 1. Browse User: Window shopping only, no purchases
# ==============================

class BrowseUser(BaseSockShopUser):
    """
    Browse User:
        Home -> Category page -> Multiple product details -> End (no login, no cart)
        Characteristics:
        - Small chance of closing page without viewing anything
        - Random number of products viewed
    """

    def on_start(self):
        self.visit_home()

    @task(5)
    def browse_flow(self):
        # ~10% chance: just open home page and end this round (very casual)
        if random.random() < 0.1:
            return

        self.visit_category_page()

        # Randomly view 1~5 detail pages, with random pauses
        for _ in range(random.randint(1, 5)):
            self.visit_detail_page_random()
            # Some chance of "getting tired" and ending early
            if random.random() < 0.15:
                break


# ==============================
# 2. Register User: Register + Login + Add to cart (mostly no checkout)
# ==============================

class RegisterUser(BaseSockShopUser):
    """
    Register User:
        Home -> Register -> Login -> Category -> Detail -> Add to cart -> View basket
        - Most cases don't checkout, just "trying out"
        - Small chance of actually checking out
    """

    def on_start(self):
        self.visit_home()
        self.register_new_user()
        if self.account:
            self.login_existing_user()

    @task(3)
    def register_and_browse(self):
        self.visit_category_page()
        pid = self.visit_detail_page_random()
        # View 0~2 more products
        for _ in range(random.randint(0, 2)):
            self.visit_detail_page_random()

        # 70% chance to only add to cart, not too many items
        if random.random() < 0.7:
            for _ in range(random.randint(1, 2)):
                self.add_to_cart(pid)
        else:
            # Remaining 30% cases, just view details, don't add
            pass

        self.visit_basket_page()

        # Only 20% chance to actually checkout, most new users just try
        if random.random() < 0.2:
            self.checkout(expect_success=True, execute_prob=0.8)


# ==============================
# 3. Decision User: Login + Compare + Add to cart + Checkout
# ==============================

class DecisionUser(BaseSockShopUser):
    """
    Decision User:
        Login -> Category -> Multiple detail views (compare/read reviews) -> Add to cart -> Checkout
        - Still some chance of abandoning during comparison (no add to cart, no checkout)
    """

    def on_start(self):
        self.visit_home()
        # Use one random account from the 20 existing accounts
        self.login_existing_user()

    @task(4)
    def decision_flow(self):
        self.visit_category_page()

        # View details multiple times, simulating comparison and review reading
        viewed_pids = []
        for _ in range(random.randint(2, 5)):
            pid = self.visit_detail_page_random()
            viewed_pids.append(pid)
            # Each view has some chance of "leaving after viewing", not continuing
            if random.random() < 0.1:
                # Just browse, don't buy
                return

        # ~20% chance of just comparing prices, not adding to cart
        if random.random() < 0.2:
            return

        # Pick some products from viewed list to add to cart (random quantity)
        for pid in random.sample(viewed_pids, min(len(viewed_pids), random.randint(1, 3))):
            self.add_to_cart(pid)

        self.visit_basket_page()

        # Decision users have high purchase probability, e.g. 80%
        self.checkout(expect_success=True, execute_prob=0.8)
        # After successful order, most will check order page, but may close directly
        if random.random() < 0.7:
            self.visit_orders_page()


# ==============================
# 4. Multi-Purchase User: Multiple orders
# ==============================

class MultiPurchaseUser(BaseSockShopUser):
    """
    Multi-Purchase User:
        Login -> First add to cart + checkout -> Add again + checkout again -> View orders
        - Second order has some chance of being abandoned
    """

    def on_start(self):
        self.visit_home()
        self.login_existing_user()

    @task(2)
    def multi_purchase_flow(self):
        # First order
        pids = []
        for _ in range(random.randint(1, 3)):
            pid = self.visit_detail_page_random()
            pids.append(pid)
            self.add_to_cart(pid)

        self.visit_basket_page()
        self.checkout(expect_success=True, execute_prob=0.9)

        # Second order (same session), but doesn't always happen
        if random.random() < 0.6:  # 60% of requests proceed to second order
            pids2 = []
            for _ in range(random.randint(1, 2)):
                pid = self.visit_detail_page_random()
                pids2.append(pid)
                self.add_to_cart(pid)
            self.visit_basket_page()
            # Second order has slightly lower execution probability
            self.checkout(expect_success=True, execute_prob=0.7)

        # Orders page is also only visited with some probability
        if random.random() < 0.8:
            self.visit_orders_page()


# ==============================
# 5. Discount User: Search for deals + Hesitate to buy
# ==============================

class DiscountUser(BaseSockShopUser):
    """
    Discount User:
        Login -> Category -> View details -> Add to cart -> Maybe checkout, maybe not
        - Reinforced probability logic to reflect "hesitant user"
    """

    def on_start(self):
        self.visit_home()
        self.login_existing_user()

    @task(1)
    def discount_flow(self):
        self.visit_category_page()
        viewed_pids = []
        for _ in range(random.randint(2, 4)):
            pid = self.visit_detail_page_random()
            viewed_pids.append(pid)

        # 50% chance to add some "viewed products" to cart
        if random.random() < 0.5:
            for pid in random.sample(viewed_pids, min(len(viewed_pids), random.randint(1, 2))):
                self.add_to_cart(pid)

        self.visit_basket_page()

        # Discount user: 60% chance to checkout (but checkout itself has 70% execution probability)
        if random.random() < 0.6:
            self.checkout(expect_success=True, execute_prob=0.7)
            if random.random() < 0.7:
                self.visit_orders_page()


