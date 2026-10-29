"""
End-to-End Automated Verification Script for Enterprise Document Intelligence RAG Engine.
Validates:
1. DB Initialization & Admin Seeding (nagahanuma / Rag@Prod)
2. User Registration & Strict Database-Level Uniqueness (No duplicate username, No duplicate email)
3. User Authentication (Login / Password Verification)
4. User BYOK Custom API Key Configuration
5. User-Specific Document Ownership & Multi-Tenant Query Filtering
6. Chat Conversational History Auto-Trimming (Max 30 Messages)
7. 90-Day Inactivity Account & Document Retention Purge
8. PROD_MODE Gatekeeper (PROD_MODE='Y' rejects local requests; PROD_MODE='N' allows them)
"""

import os
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.config import settings
from src.db import (
    init_db,
    seed_admin_user,
    register_user,
    authenticate_user,
    get_user_by_id,
    update_user_api_keys,
    get_all_documents,
    insert_document,
    delete_document,
    save_chat_message,
    get_recent_chat_history,
    cleanup_inactive_users,
    get_db_connection,
    is_postgres
)
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

passed_tests = 0
failed_tests = 0

def report(name: str, passed: bool, msg: str = ""):
    global passed_tests, failed_tests
    if passed:
        passed_tests += 1
        print(f"  ✅ [PASS] {name} {f'({msg})' if msg else ''}")
    else:
        failed_tests += 1
        print(f"  ❌ [FAIL] {name} -> {msg}")


def test_1_db_init_and_admin_seeding():
    print("\n--- Test 1: Database Initialization & Admin Seeding ---")
    try:
        init_db()
        admin = authenticate_user("nagahanuma", "Rag@Prod")
        report("Admin Seeding", admin is not None and admin.get("role") == "admin", f"Role: {admin.get('role') if admin else 'None'}")
    except Exception as e:
        report("Admin Seeding", False, str(e))


def test_2_registration_uniqueness_validation():
    print("\n--- Test 2: User Registration & Duplicate Validation ---")
    test_user = f"testuser_{uuid.uuid4().hex[:6]}"
    test_email = f"{test_user}@testcorp.ai"
    test_pwd = "SecurePassword123"

    try:
        # Step 2a: Register new user
        user = register_user(test_user, test_email, test_pwd)
        report("New User Registration", user["username"] == test_user and user["email"] == test_email)

        # Step 2b: Duplicate Username rejection
        dup_username_rejected = False
        try:
            register_user(test_user, f"different_{uuid.uuid4().hex[:4]}@testcorp.ai", "SomePass123")
        except ValueError as e:
            dup_username_rejected = True
            msg = str(e)
        report("Duplicate Username Rejection", dup_username_rejected, "Blocked duplicate username")

        # Step 2c: Duplicate Email rejection
        dup_email_rejected = False
        try:
            register_user(f"diff_user_{uuid.uuid4().hex[:4]}", test_email, "SomePass123")
        except ValueError as e:
            dup_email_rejected = True
            msg = str(e)
        report("Duplicate Email Rejection", dup_email_rejected, "Blocked duplicate email")

        # Step 2d: Test via FastAPI HTTP endpoint
        res_dup = client.post("/api/v1/auth/register", json={
            "username": test_user,
            "email": f"another_{uuid.uuid4().hex[:4]}@testcorp.ai",
            "password": "Password123"
        })
        report("FastAPI Register Endpoint 400 Bad Request", res_dup.status_code == 400, f"HTTP {res_dup.status_code}")

    except Exception as e:
        report("Registration Uniqueness Validation", False, str(e))


def test_3_authentication():
    print("\n--- Test 3: User Authentication (Login) ---")
    try:
        # Valid login
        admin = authenticate_user("nagahanuma", "Rag@Prod")
        report("Valid Login Authentication", admin is not None)

        # Invalid password
        bad_pwd = authenticate_user("nagahanuma", "WrongPassword")
        report("Invalid Password Rejection", bad_pwd is None)

        # Non-existent user
        no_user = authenticate_user("nonexistent_user_999", "Pass123")
        report("Non-existent User Rejection", no_user is None)
    except Exception as e:
        report("Authentication", False, str(e))


def test_4_user_byok_settings():
    print("\n--- Test 4: User BYOK Custom API Key Configuration ---")
    test_u = f"byok_{uuid.uuid4().hex[:6]}"
    user = register_user(test_u, f"{test_u}@corp.com", "Pass123")
    user_id = user["user_id"]

    try:
        # Update custom keys via endpoint
        res = client.post(f"/api/v1/user/settings?user_id={user_id}", json={
            "custom_groq_key": "gsk_custom_user_groq_key",
            "custom_openai_key": "sk-custom-user-openai-key"
        })
        report("BYOK Settings Update API", res.status_code == 200)

        # Verify persisted values
        updated_u = get_user_by_id(user_id)
        report(
            "BYOK Keys Persisted in DB",
            updated_u["custom_groq_key"] == "gsk_custom_user_groq_key" and
            updated_u["custom_openai_key"] == "sk-custom-user-openai-key"
        )
    except Exception as e:
        report("User BYOK Settings", False, str(e))


def test_5_document_user_isolation():
    print("\n--- Test 5: Multi-Tenant Document Ownership & Isolation ---")
    user_a = register_user(f"usera_{uuid.uuid4().hex[:5]}", f"usera_{uuid.uuid4().hex[:5]}@test.com", "Pass123")
    user_b = register_user(f"userb_{uuid.uuid4().hex[:5]}", f"userb_{uuid.uuid4().hex[:5]}@test.com", "Pass123")

    doc_a_id = f"doc_{uuid.uuid4().hex[:8]}"
    doc_b_id = f"doc_{uuid.uuid4().hex[:8]}"

    try:
        # Insert Doc A owned by User A
        insert_document(doc_a_id, "UserA_Confidential_Policy.pdf", f"hash_{doc_a_id}", 5, 20480, user_id=user_a["user_id"])
        # Insert Doc B owned by User B
        insert_document(doc_b_id, "UserB_Operations_Manual.pdf", f"hash_{doc_b_id}", 10, 40960, user_id=user_b["user_id"])

        # User A should only see Doc A
        docs_a = get_all_documents(user_id=user_a["user_id"])
        user_a_sees_only_own = any(d["doc_id"] == doc_a_id for d in docs_a) and not any(d["doc_id"] == doc_b_id for d in docs_a)
        report("User A Document Isolation", user_a_sees_only_own, f"User A sees {len(docs_a)} documents")

        # User B should only see Doc B
        docs_b = get_all_documents(user_id=user_b["user_id"])
        user_b_sees_only_own = any(d["doc_id"] == doc_b_id for d in docs_b) and not any(d["doc_id"] == doc_a_id for d in docs_b)
        report("User B Document Isolation", user_b_sees_only_own, f"User B sees {len(docs_b)} documents")

        # Admin sees all documents
        admin_docs = get_all_documents()
        admin_sees_all = any(d["doc_id"] == doc_a_id for d in admin_docs) and any(d["doc_id"] == doc_b_id for d in admin_docs)
        report("Admin Global Visibility", admin_sees_all, f"Admin sees total {len(admin_docs)} documents")

        # Cleanup test docs
        delete_document(doc_a_id)
        delete_document(doc_b_id)
    except Exception as e:
        report("Document Isolation", False, str(e))


def test_6_chat_history_capping():
    print("\n--- Test 6: Chat Conversational History Auto-Trimming (Max 30) ---")
    test_session = f"sess_test_{uuid.uuid4().hex[:8]}"
    try:
        # Insert 35 messages
        for i in range(1, 36):
            role = "user" if i % 2 != 0 else "assistant"
            save_chat_message(test_session, role, f"Message number {i}")

        history = get_recent_chat_history(test_session, limit=100)
        report(
            "Max 30 Messages Auto-Trimming",
            len(history) <= 30,
            f"Expected <=30 messages, got: {len(history)} messages. Oldest pruned: {history[0]['content']}"
        )
    except Exception as e:
        report("Chat History Capping", False, str(e))


def test_7_inactive_users_retention_purge():
    print("\n--- Test 7: 90-Day Inactivity Auto-Cleanup ---")
    inactive_user = register_user(f"inactive_{uuid.uuid4().hex[:5]}", f"inactive_{uuid.uuid4().hex[:5]}@test.com", "Pass123")
    inactive_id = inactive_user["user_id"]

    try:
        # Simulate inactive date > 95 days ago
        with get_db_connection() as conn:
            if is_postgres():
                with conn.cursor() as cur:
                    cur.execute("UPDATE users SET last_active_at = NOW() - INTERVAL '100 days' WHERE user_id = %s;", (inactive_id,))
            else:
                conn.execute("UPDATE users SET last_active_at = datetime('now', '-100 days') WHERE user_id = ?", (inactive_id,))

        # Run cleanup
        purge_res = cleanup_inactive_users(days_inactive=90)
        user_purged = get_user_by_id(inactive_id) is None
        admin_safe = authenticate_user("nagahanuma", "Rag@Prod") is not None

        report("Inactive User Purged (>90 Days)", user_purged and purge_res["purged_users"] >= 1, f"Purged: {purge_res['purged_users']} users")
        report("Admin Account Exempt from Purge", admin_safe, "nagahanuma is safe")
    except Exception as e:
        report("Inactivity Retention Purge", False, str(e))


def test_8_prod_mode_gatekeeper():
    print("\n--- Test 8: PROD_MODE Gatekeeper (PROD_MODE='Y' vs 'N') ---")
    try:
        # Case A: PROD_MODE = 'N' (Accepts local requests)
        settings.PROD_MODE = "N"
        res_local = client.get("/health", headers={"host": "localhost:8000"})
        report("PROD_MODE='N' Allows Local Request", res_local.status_code == 200, f"HTTP {res_local.status_code}")

        # Case B: PROD_MODE = 'Y' (Rejects local requests from localhost / 127.0.0.1)
        settings.PROD_MODE = "Y"
        res_blocked = client.get("/health", headers={"host": "localhost:8000"})
        report("PROD_MODE='Y' Blocks Local Request", res_blocked.status_code == 403, f"HTTP {res_blocked.status_code} ({res_blocked.json().get('detail', '')})")

        # Reset back to 'N' for local development
        settings.PROD_MODE = "N"
        res_reset = client.get("/health", headers={"host": "localhost:8000"})
        report("Reset PROD_MODE='N' Resumes Local Operation", res_reset.status_code == 200)

    except Exception as e:
        report("PROD_MODE Gatekeeper", False, str(e))


if __name__ == "__main__":
    print("=" * 65)
    print("🚀 ENTERPRISE DOCUMENT INTELLIGENCE RAG ENGINE - SYSTEM VALIDATION")
    print("=" * 65)

    test_1_db_init_and_admin_seeding()
    test_2_registration_uniqueness_validation()
    test_3_authentication()
    test_4_user_byok_settings()
    test_5_document_user_isolation()
    test_6_chat_history_capping()
    test_7_inactive_users_retention_purge()
    test_8_prod_mode_gatekeeper()

    print("\n" + "=" * 65)
    print(f"📊 SUMMARY: {passed_tests} PASSED, {failed_tests} FAILED")
    print("=" * 65)

    if failed_tests > 0:
        sys.exit(1)
    else:
        sys.exit(0)
