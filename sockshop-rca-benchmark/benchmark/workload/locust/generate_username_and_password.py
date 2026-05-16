import random
import string
import json

# ===== Reproducible random seed (recommended to keep fixed in experiments) =====
random.seed(42)

# Reusable word pools
prefix_words = [
    "nova", "pixel", "orbit", "luna", "zen", "swift", "mint",
    "alpha", "echo", "prime", "flux", "stellar", "astro",
    "solar", "neon", "quantum", "apex", "omega", "vortex",
    "blaze", "crystal", "storm", "ridge", "pulse", "glow"
]

middle_words = [
    "User", "Cart", "Buyer", "Shopper", "Pilot", "Rider",
    "Shop", "Cloud", "Deal", "Market", "Store", "Wave"
]

def random_password(length=10):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def generate_unique_users(n):
    users = []
    usernames = set()

    while len(users) < n:
        prefix = random.choice(prefix_words)
        middle = random.choice(middle_words)
        number = random.randint(10, 999)

        username = f"{prefix}{middle}{number}"

        if username not in usernames:
            usernames.add(username)
            users.append({
                "username": username,
                "password": random_password()
            })

    return users


# ===== Generate 800 users =====
new_users = generate_unique_users(800)

with open("random_users_800.json", "w") as f:
    json.dump(new_users, f, indent=4)

print("Generated:", len(new_users))
