# pharmalink_streamlit_demo.py
# Streamlit demo app for "PharmaLink" – an all-in-one pharmacy inventory + POS + insights tool.
# How to run:
#   1) Install deps: pip install streamlit pandas numpy scikit-learn python-dateutil
#   2) Run: streamlit run pharmalink_streamlit_demo.py
# Notes:
#   - This is a demo with in-memory data. In production, replace with a real DB and auth.
#   - "NAFDAC check", supplier links, WhatsApp, etc. are simulated to illustrate UX flows.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="PharmaLink Demo", page_icon="🧪", layout="wide")

# ---------------------------
# Utilities
# ---------------------------
def money(x):
    try:
        return f"₦{float(x):,.2f}"
    except Exception:
        return x

def generate_nafdac_no(idx):
    return f"NAFDAC-{100000 + idx}"

def random_expiry(start_days=90, end_days=720, n=1, seed=None):
    rng = np.random.default_rng(seed)
    return [date.today() + timedelta(days=int(x)) for x in rng.integers(start_days, end_days, n)]

def simple_forecast(series: pd.Series, horizon_days=30):
    """
    Very lightweight forecaster: linear trend + month seasonality (OLS on features).
    Expects a daily series indexed by date with numeric values.
    """
    if series.isna().all() or series.sum() == 0:
        # no signal, return zeros
        idx = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        return pd.Series([0]*horizon_days, index=idx)

    df = series.reset_index()
    df.columns = ["ds", "y"]
    df["t"] = (df["ds"] - df["ds"].min()).dt.days
    df["month"] = df["ds"].dt.month

    # One-hot encode month
    month_dummies = pd.get_dummies(df["month"], prefix="m", drop_first=True)
    X = pd.concat([df[["t"]], month_dummies], axis=1)
    y = df["y"].values

    model = LinearRegression()
    model.fit(X, y)

    # Future
    last_day = df["ds"].max()
    future_idx = pd.date_range(last_day + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    fut_t = (future_idx - df["ds"].min()).days
    fut_month = future_idx.month
    fut_month_dummies = pd.get_dummies(fut_month, prefix="m", drop_first=True)

    # align columns
    fut_X = pd.DataFrame({"t": fut_t})
    for col in X.columns:
        if col.startswith("m_"):
            fut_X[col] = 0
    for col in fut_month_dummies.columns:
        if col in fut_X.columns:
            fut_X[col] = fut_month_dummies[col]

    yhat = model.predict(fut_X[X.columns])
    yhat = np.clip(yhat, a_min=0, a_max=None)
    return pd.Series(yhat, index=future_idx)

def badge(text, color="#2563eb"):
    return f"<span style='padding:2px 6px;border-radius:8px;background-color:{color};color:white;font-size:0.75rem'>{text}</span>"

def init_state():
    if "bootstrapped" not in st.session_state:
        # ---------------------------
        # Seed demo data
        # ---------------------------
        rng = np.random.default_rng(42)
        branches = ["Lagos - Ikeja", "Abuja - Wuse", "Port Harcourt - GRA"]
        st.session_state.branches = branches

        # Drug catalog
        names = [
            ("Paracetamol", "Tablet", "500mg", "Pain Relief", "Emzor Pharma"),
            ("Amoxicillin", "Capsule", "500mg", "Antibiotic", "Fidson"),
            ("Artemether/Lumefantrine", "Tablet", "20/120mg", "Antimalarial", "GSK"),
            ("Ibuprofen", "Tablet", "400mg", "Pain Relief", "Sanofi"),
            ("Cough Syrup", "Syrup", "100ml", "Cold & Flu", "May & Baker"),
            ("Vitamin C", "Tablet", "1000mg", "Vitamins", "Emzor Pharma"),
            ("Metformin", "Tablet", "500mg", "Diabetes", "Sun Pharma"),
            ("Lisinopril", "Tablet", "10mg", "Hypertension", "Pfizer"),
            ("Cetirizine", "Tablet", "10mg", "Allergy", "GSK"),
            ("ORS", "Sachet", "27.9g", "Rehydration", "Unilever"),
        ]

        rows = []
        for i, (n, form, dose, category, company) in enumerate(names, start=1):
            barcode = f"893000{i:04d}"
            nafdac = generate_nafdac_no(i)
            price = float(rng.integers(400, 6000))
            stock = int(rng.integers(20, 250))
            reorder = int(rng.integers(10, 60))
            exp = random_expiry(n=1, seed=i)[0]
            rows.append({
                "drug_id": i,
                "name": n,
                "form": form,
                "dosage": dose,
                "category": category,
                "company": company,
                "nafdac_no": nafdac,
                "barcode": barcode,
                "price": price,
                "stock": stock,
                "reorder_level": reorder,
                "expiry_date": exp,
                "branch": rng.choice(branches)
            })
        st.session_state.inventory = pd.DataFrame(rows)

        # Suppliers
        st.session_state.suppliers = pd.DataFrame([
            {"supplier_id": 1, "name": "MedPlus Distributors", "contact": "+234800000001", "email": "sales@medplus.ng"},
            {"supplier_id": 2, "name": "HealthCo Wholesale", "contact": "+234800000002", "email": "orders@healthco.ng"},
            {"supplier_id": 3, "name": "NaijaPharm Supply", "contact": "+234800000003", "email": "hello@naijapharm.ng"},
        ])

        # Customers (with credit support)
        st.session_state.customers = pd.DataFrame([
            {"customer_id": 1, "name": "John Doe", "phone": "+2348100000001", "credit_limit": 20000.0, "credit_balance": 0.0},
            {"customer_id": 2, "name": "Ada Obi", "phone": "+2348100000002", "credit_limit": 50000.0, "credit_balance": 15000.0},
            {"customer_id": 3, "name": "Ibrahim Musa", "phone": "+2348100000003", "credit_limit": 30000.0, "credit_balance": 0.0},
        ])

        # Historical daily sales for 18 months by drug (simulate seasonality: higher malaria during rainy months Apr-Oct)
        start_date = (date.today().replace(day=1) - relativedelta(months=17))
        dates = pd.date_range(start_date, date.today(), freq="D")

        sales_rows = []
        for _, item in st.session_state.inventory.iterrows():
            base = np.random.default_rng(item.drug_id).integers(1, 6)  # base daily units
            for d in dates:
                seasonal = 1.0
                if item["category"] == "Antimalarial":
                    if 4 <= d.month <= 10:
                        seasonal = 1.6
                elif item["category"] in ("Cold & Flu", "Allergy"):
                    if d.month in (6,7,8,9):  # rainy/cold season peaks
                        seasonal = 1.3
                noise = np.random.default_rng(int(d.strftime("%Y%m%d")) + item.drug_id).normal(0, 0.8)
                qty = max(0, int(round(base * seasonal + noise)))
                if qty > 0:
                    sales_rows.append({
                        "date": d.date(),
                        "drug_id": item["drug_id"],
                        "qty": qty,
                        "unit_price": item["price"],
                        "branch": item["branch"],
                        "payment_method": np.random.choice(["Cash", "Card", "Transfer", "USSD", "Wallet"]),
                    })
        st.session_state.sales = pd.DataFrame(sales_rows)

        # Cart, user role
        st.session_state.cart = []
        st.session_state.role = "Cashier"
        st.session_state.active_branch = branches[0]
        st.session_state.bootstrapped = True

def filter_branch(df):
    return df[df["branch"] == st.session_state.active_branch] if "branch" in df.columns else df

def nafdac_check(nafdac_no: str) -> bool:
    # Demo: accept NAFDAC numbers in our catalog; reject unknown format
    inv = st.session_state.inventory
    if nafdac_no in set(inv["nafdac_no"]):
        return True
    return nafdac_no.startswith("NAFDAC-") and nafdac_no[7:].isdigit()

def whatsapp_link(phone: str, message: str) -> str:
    # Phone must be digits. This is a simple formatter; users may adapt for their locale formatting.
    clean = "".join([c for c in phone if c.isdigit()])
    import urllib.parse as up
    return f"https://wa.me/{clean}?text=" + up.quote(message)

# ---------------------------
# UI: Sidebar
# ---------------------------
init_state()

with st.sidebar:
    st.title("🧪 PharmaLink Demo")
    st.caption("All-in-one: Inventory • POS • Insights")
    st.selectbox("Branch", st.session_state.branches, key="active_branch")
    st.selectbox("Role", ["Owner/Manager", "Pharmacist", "Cashier"], key="role")
    page = st.radio("Go to", [
        "🏠 Dashboard",
        "🧾 Point of Sale",
        "📦 Inventory",
        "👥 Customers & Credit",
        "🚚 Suppliers & Restock",
        "📈 Forecasts & Recommendations",
        "📑 Reports",
        "⚙️ Settings"
    ])

# ---------------------------
# Page: Dashboard
# ---------------------------
def page_dashboard():
    st.subheader(f"Branch: {st.session_state.active_branch} — Overview")
    inv = filter_branch(st.session_state.inventory)
    sales = filter_branch(st.session_state.sales)

    # KPIs
    today = date.today()
    sales_today = sales[sales["date"] == today]
    revenue_today = (sales_today["qty"] * sales_today["unit_price"]).sum()

    last_30 = date.today() - timedelta(days=30)
    sales_30 = sales[sales["date"] >= last_30]
    revenue_30 = (sales_30["qty"] * sales_30["unit_price"]).sum()

    low_stock = inv[inv["stock"] <= inv["reorder_level"]].shape[0]
    near_expiry = inv[inv["expiry_date"] <= (today + timedelta(days=60))].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue (Today)", money(revenue_today))
    c2.metric("Revenue (Last 30 days)", money(revenue_30))
    c3.metric("Low Stock Items", int(low_stock))
    c4.metric("Near-Expiry (≤60 days)", int(near_expiry))

    # Sales trend
    st.markdown("### Sales Trend (Last 90 days)")
    trend = sales[sales["date"] >= (today - timedelta(days=90))].copy()
    if not trend.empty:
        trend_daily = trend.groupby("date").apply(lambda x: (x["qty"] * x["unit_price"]).sum()).reset_index(name="revenue")
        trend_daily = trend_daily.sort_values("date")
        trend_daily = trend_daily.set_index("date")
        st.line_chart(trend_daily)
    else:
        st.info("No recent sales data.")

    # Fast movers & Revenue by category
    st.markdown("### Fast Movers & Category Mix")
    c1, c2 = st.columns(2)

    with c1:
        fm = sales_30.groupby("drug_id")["qty"].sum().reset_index().sort_values("qty", ascending=False).head(10)
        fm = fm.merge(inv[["drug_id", "name"]], on="drug_id", how="left")
        st.dataframe(fm.rename(columns={"name":"Drug", "qty":"Units sold (30d)"}), use_container_width=True)

    with c2:
        cat = sales_30.merge(inv[["drug_id", "category"]], on="drug_id", how="left")
        if not cat.empty:
            mix = cat.groupby("category").apply(lambda x: (x["qty"] * x["unit_price"]).sum()).reset_index(name="revenue")
            mix = mix.sort_values("revenue", ascending=False)
            st.dataframe(mix, use_container_width=True)
        else:
            st.info("No category data.")

# ---------------------------
# Page: POS
# ---------------------------
def page_pos():
    st.subheader("Point of Sale")
    inv = filter_branch(st.session_state.inventory).copy()
    inv["label"] = inv["name"] + " • " + inv["dosage"] + " • " + inv["form"] + " — " + inv["company"]

    c1, c2 = st.columns([2,1])
    with c1:
        choice = st.selectbox("Search drug", inv["label"].tolist())
        selected = inv[inv["label"] == choice].iloc[0]
        qty = st.number_input("Quantity", min_value=1, max_value=int(selected["stock"]), value=1)
        payment = st.selectbox("Payment method", ["Cash", "Card", "Transfer", "USSD", "Wallet", "On Credit (Customer)"])
        if payment == "On Credit (Customer)":
            customers = st.session_state.customers
            cust_name = st.selectbox("Select customer", customers["name"].tolist())
            customer = customers[customers["name"] == cust_name].iloc[0]
        else:
            customer = None

        if st.button("Add to cart"):
            st.session_state.cart.append({
                "drug_id": int(selected["drug_id"]),
                "name": selected["name"],
                "qty": int(qty),
                "unit_price": float(selected["price"]),
                "payment": payment,
                "customer_id": int(customer["customer_id"]) if customer is not None else None,
                "branch": st.session_state.active_branch
            })
            st.success(f"Added {qty} x {selected['name']} to cart.")

    with c2:
        st.markdown("#### Quick Scan")
        barcode = st.text_input("Enter / scan barcode")
        if st.button("Add by barcode"):
            match = inv[inv["barcode"] == barcode]
            if match.empty:
                st.error("Barcode not found for this branch.")
            else:
                row = match.iloc[0]
                st.session_state.cart.append({
                    "drug_id": int(row["drug_id"]),
                    "name": row["name"],
                    "qty": 1,
                    "unit_price": float(row["price"]),
                    "payment": "Cash",
                    "customer_id": None,
                    "branch": st.session_state.active_branch
                })
                st.success(f"Added 1 x {row['name']} by barcode.")

    st.divider()
    st.markdown("### Cart")
    if st.session_state.cart:
        cart_df = pd.DataFrame(st.session_state.cart)
        cart_df["line_total"] = cart_df["qty"] * cart_df["unit_price"]
        st.dataframe(cart_df[["name","qty","unit_price","payment","line_total"]], use_container_width=True)
        total = cart_df["line_total"].sum()
        st.metric("Total", money(total))

        if st.button("Checkout"):
            # validate stock
            inv = st.session_state.inventory
            ok = True
            for item in st.session_state.cart:
                stock = int(inv.loc[inv["drug_id"] == item["drug_id"], "stock"].iloc[0])
                if item["qty"] > stock:
                    st.error(f"Insufficient stock for {item['name']} (have {stock}, need {item['qty']}).")
                    ok = False
            if ok:
                # update sales, inventory, and customer credit if needed
                for item in st.session_state.cart:
                    # inventory decrement
                    inv_idx = inv[inv["drug_id"] == item["drug_id"]].index
                    inv.loc[inv_idx, "stock"] = inv.loc[inv_idx, "stock"] - item["qty"]

                    # sales append
                    st.session_state.sales = pd.concat([st.session_state.sales, pd.DataFrame([{
                        "date": date.today(),
                        "drug_id": item["drug_id"],
                        "qty": item["qty"],
                        "unit_price": item["unit_price"],
                        "branch": item["branch"],
                        "payment_method": item["payment"]
                    }])], ignore_index=True)

                    # credit handling
                    if item["payment"] == "On Credit (Customer)" and item["customer_id"] is not None:
                        amount = item["qty"] * item["unit_price"]
                        cust_idx = st.session_state.customers[st.session_state.customers["customer_id"] == item["customer_id"]].index
                        st.session_state.customers.loc[cust_idx, "credit_balance"] += amount

                st.session_state.inventory = inv
                st.success("Checkout complete ✅")
                st.session_state.cart = []
    else:
        st.info("Cart is empty.")

# ---------------------------
# Page: Inventory
# ---------------------------
def page_inventory():
    st.subheader("Inventory")
    inv = filter_branch(st.session_state.inventory)

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        q = st.text_input("Search by name/company/dosage")
    with c2:
        cat = st.selectbox("Filter by category", ["All"] + sorted(inv["category"].unique().tolist()))
    with c3:
        only_low = st.checkbox("Show only low stock (≤ reorder level)")

    df = inv.copy()
    if q:
        mask = (
            df["name"].str.contains(q, case=False) |
            df["company"].str.contains(q, case=False) |
            df["dosage"].str.contains(q, case=False)
        )
        df = df[mask]
    if cat != "All":
        df = df[df["category"] == cat]
    if only_low:
        df = df[df["stock"] <= df["reorder_level"]]

    # Badges
    def row_badges(row):
        b = []
        if row["stock"] <= row["reorder_level"]:
            b.append(badge("LOW", "#e76f51"))
        if row["expiry_date"] <= (date.today() + timedelta(days=60)):
            b.append(badge("EXPIRING", "#e9c46a"))
        return " ".join(b)

    df = df.assign(flags=df.apply(row_badges, axis=1))
    st.dataframe(df[["drug_id", "name", "dosage", "form", "company", "category",
                     "nafdac_no", "barcode", "price", "stock", "reorder_level",
                     "expiry_date", "flags"]], use_container_width=True)

    st.markdown("#### Add / Edit Item")
    with st.form("edit_item"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            name = st.text_input("Name")
            dosage = st.text_input("Dosage (e.g., 500mg)")
        with c2:
            form = st.selectbox("Form", ["Tablet", "Capsule", "Syrup", "Sachet", "Injection", "Ointment"])
            category = st.selectbox("Category", sorted(st.session_state.inventory["category"].unique().tolist()))
        with c3:
            company = st.text_input("Company")
            price = st.number_input("Unit Price (₦)", min_value=0.0, value=1000.0, step=50.0)
        with c4:
            stock = st.number_input("Stock", min_value=0, value=10, step=1)
            reorder = st.number_input("Reorder Level", min_value=0, value=5, step=1)

        colx, coly = st.columns(2)
        with colx:
            nafdac_no = st.text_input("NAFDAC No.", value=generate_nafdac_no(9999))
        with coly:
            barcode = st.text_input("Barcode", value=f"893000{np.random.default_rng().integers(1000,9999)}")
        exp = st.date_input("Expiry Date", value=date.today() + timedelta(days=365))

        submitted = st.form_submit_button("Save Item")
        if submitted:
            new_id = int(st.session_state.inventory["drug_id"].max()) + 1
            st.session_state.inventory = pd.concat([st.session_state.inventory, pd.DataFrame([{
                "drug_id": new_id, "name": name, "form": form, "dosage": dosage,
                "category": category, "company": company, "nafdac_no": nafdac_no,
                "barcode": barcode, "price": float(price), "stock": int(stock),
                "reorder_level": int(reorder), "expiry_date": exp, "branch": st.session_state.active_branch
            }])], ignore_index=True)
            st.success(f"Saved {name} [{dosage}]")

    st.markdown("#### Verify NAFDAC")
    code = st.text_input("Enter NAFDAC number to verify", key="nafdac_check")
    if st.button("Run NAFDAC Check"):
        if nafdac_check(code):
            st.success("NAFDAC number appears valid/authentic ✅ (demo check)")
        else:
            st.error("NAFDAC number could not be verified ❌ (demo check)")

# ---------------------------
# Page: Customers & Credit
# ---------------------------
def page_customers():
    st.subheader("Customers & Credit")
    df = st.session_state.customers.copy()
    df["available_credit"] = df["credit_limit"] - df["credit_balance"]
    st.dataframe(df, use_container_width=True)

    st.markdown("#### Adjust Credit")
    names = df["name"].tolist()
    cust = st.selectbox("Customer", names)
    action = st.selectbox("Action", ["Record Repayment", "Increase Limit", "Decrease Limit"])
    amt = st.number_input("Amount (₦)", min_value=0.0, step=500.0, value=1000.0)
    if st.button("Apply"):
        idx = st.session_state.customers[st.session_state.customers["name"] == cust].index
        if action == "Record Repayment":
            st.session_state.customers.loc[idx, "credit_balance"] = np.maximum(
                0.0, st.session_state.customers.loc[idx, "credit_balance"] - amt
            )
        elif action == "Increase Limit":
            st.session_state.customers.loc[idx, "credit_limit"] += amt
        else:
            st.session_state.customers.loc[idx, "credit_limit"] = np.maximum(
                0.0, st.session_state.customers.loc[idx, "credit_limit"] - amt
            )
        st.success("Updated.")

    st.markdown("#### WhatsApp Invoice (demo)")
    cust2 = st.selectbox("Select customer for WhatsApp", names, key="wa_cust")
    selected = st.session_state.customers[st.session_state.customers["name"] == cust2].iloc[0]
    sample_msg = f"Hello {selected['name']}, your invoice from PharmaLink is ₦5,000. Thank you."
    wa = whatsapp_link(selected["phone"], sample_msg)
    st.write(f"[Open WhatsApp message]({wa})")

# ---------------------------
# Page: Suppliers & Restock
# ---------------------------
def page_suppliers():
    st.subheader("Suppliers & Restock")
    st.dataframe(st.session_state.suppliers, use_container_width=True)

    st.markdown("#### Restock Recommendations")
    inv = filter_branch(st.session_state.inventory)
    low = inv[inv["stock"] <= inv["reorder_level"]].copy()
    if low.empty:
        st.success("No low-stock items. 👍")
    else:
        low["suggested_order_qty"] = (low["reorder_level"] * 2 - low["stock"]).clip(lower=1)
        st.dataframe(low[["name","dosage","form","company","stock","reorder_level","suggested_order_qty"]], use_container_width=True)

        csv = low.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV for Supplier", data=csv, file_name="restock_recommendations.csv", mime="text/csv")

# ---------------------------
# Page: Forecasts & Recommendations
# ---------------------------
def page_forecasts():
    st.subheader("Forecasts & Recommendations")
    inv = filter_branch(st.session_state.inventory)
    sales = filter_branch(st.session_state.sales)

    # Top 5 by volume last 90 days
    last_90 = date.today() - timedelta(days=90)
    s90 = sales[sales["date"] >= last_90]
    top = s90.groupby("drug_id")["qty"].sum().reset_index().sort_values("qty", ascending=False).head(5)

    if top.empty:
        st.info("Not enough sales to forecast.")
        return

    tabs = st.tabs([f"{inv.loc[inv['drug_id']==i,'name'].iloc[0]}" for i in top["drug_id"]])
    for tab, drug_id in zip(tabs, top["drug_id"]):
        with tab:
            name = inv.loc[inv["drug_id"]==drug_id, "name"].iloc[0]
            st.markdown(f"#### {name} — 30-day Forecast")
            df = sales[sales["drug_id"] == drug_id].copy()
            # build daily series
            daily = df.groupby("date")["qty"].sum().reindex(pd.date_range(df["date"].min(), df["date"].max(), freq="D"), fill_value=0)
            daily.index = pd.to_datetime(daily.index)
            fc = simple_forecast(daily, horizon_days=30)
            show = pd.concat([daily.tail(60), fc])
            show_df = pd.DataFrame({"qty": show})
            st.line_chart(show_df)

            # Recommendation logic
            # Avg daily forecast next 30d vs current stock
            avg_daily = fc.mean()
            stock = inv.loc[inv["drug_id"] == drug_id, "stock"].iloc[0]
            days_cover = (stock / avg_daily) if avg_daily > 0 else np.inf
            if avg_daily == 0:
                st.info("Low or no demand forecasted. Avoid overstocking.")
            else:
                st.write(f"Avg daily demand (next 30d): **{avg_daily:.1f}** | Current stock: **{int(stock)}** | Days of cover: **{days_cover:.1f}**")

            if avg_daily > 0 and days_cover < 15:
                need = int(max(0, np.ceil(avg_daily*30 - stock)))
                st.warning(f"⚠️ Consider reordering ~ **{need} units** to cover 30 days of demand.")

    st.markdown("### Pricing Insights (demo)")
    demo = inv[["drug_id","name","price","stock","reorder_level"]].copy()
    # Suggest small price increase for very fast movers (proxy: stock << reorder or high s90 qty)
    fast = set(top["drug_id"].tolist())
    def price_tip(row):
        if row["drug_id"] in fast:
            return "Fast mover: consider +2–5% price"
        if row["stock"] > row["reorder_level"] * 3:
            return "Slow mover: consider promo/discount"
        return "OK"
    demo["advice"] = demo.apply(price_tip, axis=1)
    st.dataframe(demo.drop(columns=["drug_id"]), use_container_width=True)

# ---------------------------
# Page: Reports
# ---------------------------
def page_reports():
    st.subheader("Reports")
    inv = filter_branch(st.session_state.inventory)
    sales = filter_branch(st.session_state.sales)

    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Start", value=date.today() - timedelta(days=30))
    with c2:
        end = st.date_input("End", value=date.today())

    mask = (sales["date"] >= start) & (sales["date"] <= end)
    report = sales[mask].copy()
    if report.empty:
        st.info("No sales in selected period.")
        return

    report["revenue"] = report["qty"] * report["unit_price"]
    detail = report.merge(inv[["drug_id","name","category","company"]], on="drug_id", how="left")
    st.markdown("#### Detailed Sales")
    st.dataframe(detail.sort_values("date"), use_container_width=True)

    st.markdown("#### Summary by Drug")
    by_drug = detail.groupby(["name"]).agg(units=("qty","sum"), revenue=("revenue","sum")).reset_index().sort_values("revenue", ascending=False)
    st.dataframe(by_drug, use_container_width=True)

    st.markdown("#### Summary by Category")
    by_cat = detail.groupby(["category"]).agg(units=("qty","sum"), revenue=("revenue","sum")).reset_index().sort_values("revenue", ascending=False)
    st.dataframe(by_cat, use_container_width=True)

    # Downloads
    st.markdown("#### Downloads")
    st.download_button("Download Detailed CSV", data=detail.to_csv(index=False).encode("utf-8"), file_name="sales_detailed.csv", mime="text/csv")
    st.download_button("Download Summary by Drug CSV", data=by_drug.to_csv(index=False).encode("utf-8"), file_name="sales_by_drug.csv", mime="text/csv")
    st.download_button("Download Summary by Category CSV", data=by_cat.to_csv(index=False).encode("utf-8"), file_name="sales_by_category.csv", mime="text/csv")

# ---------------------------
# Page: Settings
# ---------------------------
def page_settings():
    st.subheader("Settings (Demo)")
    st.markdown("- **Offline Mode**: Streamlit runs in browser; this demo stores data in memory. For true offline, use a local DB (SQLite) and background sync.")
    st.markdown("- **User Roles**: Current role limits are illustrative. Implement auth (e.g., Cognito) in production.")
    st.markdown("- **Multi-currency**: Showing ₦; add conversion APIs for more currencies.")
    st.markdown("- **Integrations**: Replace WhatsApp links and supplier exports with real APIs.")
    st.markdown("- **Counterfeit/NAFDAC**: Integrate with official verification services where available.")

# ---------------------------
# Router
# ---------------------------
if page == "🏠 Dashboard":
    page_dashboard()
elif page == "🧾 Point of Sale":
    if st.session_state.role in ("Owner/Manager", "Cashier", "Pharmacist"):
        page_pos()
    else:
        st.warning("Not permitted for your role.")
elif page == "📦 Inventory":
    if st.session_state.role in ("Owner/Manager", "Pharmacist"):
        page_inventory()
    else:
        st.warning("Not permitted for your role.")
elif page == "👥 Customers & Credit":
    if st.session_state.role in ("Owner/Manager", "Cashier"):
        page_customers()
    else:
        st.warning("Not permitted for your role.")
elif page == "🚚 Suppliers & Restock":
    if st.session_state.role in ("Owner/Manager", "Pharmacist"):
        page_suppliers()
    else:
        st.warning("Not permitted for your role.")
elif page == "📈 Forecasts & Recommendations":
    if st.session_state.role in ("Owner/Manager", "Pharmacist"):
        page_forecasts()
    else:
        st.warning("Not permitted for your role.")
elif page == "📑 Reports":
    if st.session_state.role in ("Owner/Manager", "Pharmacist"):
        page_reports()
    else:
        st.warning("Not permitted for your role.")
elif page == "⚙️ Settings":
    page_settings()
