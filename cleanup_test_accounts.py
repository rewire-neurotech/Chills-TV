# one off cleanup of load test accounts, run from the Render shell
# render dashboard, the chillstv service, Shell tab, then:
#   python cleanup_test_accounts.py
# it lists what it will delete and asks before deleting

import db

PATTERNS = ["tester%@example.com", "flagcheck%@example.com", "signinprobe%@example.com", "evidence%@example.com"]

targets = []
with db.get_conn() as conn:
    for p in PATTERNS:
        rows = conn.execute("SELECT id, email FROM users WHERE email LIKE ?", (p,)).fetchall()
        targets.extend((r["id"], r["email"]) for r in rows)

print("found", len(targets), "test accounts")
for uid, email in targets:
    print(uid, email)

if targets:
    answer = input("delete all of these? type yes: ")
    if answer.strip().lower() == "yes":
        for uid, email in targets:
            db.delete_user_account(uid)
            print("deleted", email)
        print("done,", len(targets), "removed")
    else:
        print("nothing deleted")
