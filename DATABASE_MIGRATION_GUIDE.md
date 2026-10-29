# Database Migration Guide: Supabase to AWS, Azure, or GCP

This guide provides step-by-step instructions to migrate your **Banking Compliance RAG Engine** database from Supabase to **AWS RDS / Aurora**, **Azure Database for PostgreSQL Flexible Server**, or **GCP Cloud SQL**.

All tables, indexes, and vector embeddings use standard PostgreSQL with the open-source **`pgvector`** extension, ensuring **100% portability** across cloud providers with zero vendor lock-in.

---

## 📁 Migration Artifact

The complete, ready-to-run schema script is located at:
👉 **[`schema.sql`](file:///e:/2026/ET_Hackthon/EThack/schema.sql)**

---

## 1. Cloud Provider Setup & Enabling `pgvector`

### Option A: AWS RDS / Aurora PostgreSQL (Version 15.2+)
1. **Enable pgvector in Parameter Group**:
   - In AWS RDS Console -> **Parameter Groups** -> Create or edit your custom parameter group.
   - Set `shared_preload_libraries` to include `vector` (e.g. `shared_preload_libraries = 'vector'`).
   - Attach the parameter group to your RDS/Aurora instance and reboot the instance.
2. **Execute Schema**:
   ```bash
   psql -h <your-rds-endpoint>.rds.amazonaws.com -U postgres -d postgres -f schema.sql
   ```
3. **Update Application `.env`**:
   ```ini
   DATABASE_URL=postgresql://postgres:<password>@<your-rds-endpoint>.rds.amazonaws.com:5432/postgres?sslmode=require
   ```

---

### Option B: Azure Database for PostgreSQL Flexible Server (Version 14+)
1. **Enable pgvector in Server Parameters**:
   - In Azure Portal -> Navigate to your **Azure Database for PostgreSQL Flexible Server**.
   - Go to **Settings** -> **Server parameters**.
   - Search for `azure.extensions`.
   - Add `VECTOR` to the allowed extensions list and click **Save**.
2. **Execute Schema**:
   ```bash
   psql "host=<your-server>.postgres.database.azure.com user=<admin-user> dbname=postgres sslmode=require" -f schema.sql
   ```
3. **Update Application `.env`**:
   ```ini
   DATABASE_URL=postgresql://<admin-user>:<password>@<your-server>.postgres.database.azure.com:5432/postgres?sslmode=require
   ```

---

### Option C: Google Cloud SQL (PostgreSQL 15+)
1. **Enable pgvector Flag**:
   - In GCP Console -> **Cloud SQL** -> Select your PostgreSQL instance -> **Edit configuration**.
   - Under **Database flags**, click **Add flag**.
   - Select `cloudsql.enable_pgvector` and set it to `on`.
   - Click **Save**.
2. **Execute Schema**:
   ```bash
   psql "host=<cloud-sql-public-or-private-ip> user=postgres dbname=postgres sslmode=require" -f schema.sql
   ```
3. **Update Application `.env`**:
   ```ini
   DATABASE_URL=postgresql://postgres:<password>@<cloud-sql-ip>:5432/postgres?sslmode=require
   ```

---

## 2. Moving Existing Data from Supabase to AWS / Azure / GCP

If you have existing documents and users in Supabase and want to transfer them directly:

### Step 1: Export Data from Supabase
```bash
pg_dump "postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres" \
  --no-owner --no-acl --data-only \
  -t users -t documents -t user_documents -t parent_chunks -t child_chunks -t chat_sessions -t chat_messages \
  > supabase_data_backup.sql
```

### Step 2: Initialize Tables on New Database (AWS / Azure)
```bash
psql "<new-database-connection-url>" -f schema.sql
```

### Step 3: Import Data into New Database
```bash
psql "<new-database-connection-url>" -f supabase_data_backup.sql
```

---

## 3. Automatic Table Verification & Self-Healing

Note that our application code in [`src/db.py:init_db()`](file:///e:/2026/ET_Hackthon/EThack/src/db.py#L70) is also built with **automatic table creation**:
Whenever you change `DATABASE_URL` in `.env` to a brand-new AWS or Azure database and start the server, **FastAPI will automatically create all missing tables, extensions, and indexes on boot!**

Having [`schema.sql`](file:///e:/2026/ET_Hackthon/EThack/schema.sql) allows you to pre-provision the database through Terraform, CI/CD pipelines, or enterprise DBA approvals before starting the app.
