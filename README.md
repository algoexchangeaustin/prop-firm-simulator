# 📊 Prop Firm Simulation Chatbot

A web application that lets users upload their algorithmic trading backtest results and run Monte Carlo simulations to estimate their probability of passing prop firm evaluations.

---

## 🎯 What This App Does

1. **Uploads** your backtest CSV file (trade history with P&L)
2. **Analyzes** your trading statistics (win rate, profit factor, etc.)
3. **Simulates** 200 random reshuffles of your trades against prop firm rules
4. **Reports** your probability of passing, including:
   - Pass rate percentage
   - Average days to reach profit target
   - Failure reason breakdown
   - Visual equity curves

---

## 🏢 Supported Prop Firms

- Apex Trader Funding (50K, 100K, 150K)
- Topstep (50K, 100K, 150K)
- TradeDay (50K, 100K)
- My Funded Futures (50K, 100K)
- Take Profit Trader (50K)
- Bulenox (50K)

*You can easily add more firms by editing the `prop_firms.json` file*

---

## 📁 CSV File Requirements

Your CSV file needs at minimum a **P&L column**. The app will auto-detect columns named:
- `PnL`, `pnl`, `P&L`
- `Profit`, `profit`
- `Net Profit`, `net_profit`

**Optional but recommended:** A date column (`Date`, `DateTime`, etc.) to group trades by day.

### Example CSV Format:
```csv
Date,PnL
2024-01-02,125.50
2024-01-02,-45.00
2024-01-03,300.00
2024-01-03,-150.25
```

---

# 🚀 DEPLOYMENT GUIDE (Step-by-Step for Non-Developers)

## Option 1: Streamlit Cloud (FREE & EASIEST)

This is the recommended approach. No coding required - just clicking and copy-pasting.

### Step 1: Create a GitHub Account (if you don't have one)

1. Go to [github.com](https://github.com)
2. Click "Sign Up"
3. Follow the prompts to create a free account
4. Verify your email address

### Step 2: Create a New Repository

1. Once logged into GitHub, click the **+** icon in the top right
2. Select **"New repository"**
3. Name it: `prop-firm-simulator`
4. Keep it **Public** (required for free Streamlit hosting)
5. Check **"Add a README file"**
6. Click **"Create repository"**

### Step 3: Upload the App Files

1. In your new repository, click **"Add file"** → **"Upload files"**
2. Drag and drop ALL of these files:
   - `app.py`
   - `simulation.py`
   - `prop_firms.json`
   - `requirements.txt`
   - `sample_trades.csv`
3. Scroll down and click **"Commit changes"**

### Step 4: Deploy on Streamlit Cloud

1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Click **"Sign in"** and choose **"Continue with GitHub"**
3. Authorize Streamlit to access your GitHub
4. Click **"New app"**
5. Fill in the form:
   - **Repository:** Select `your-username/prop-firm-simulator`
   - **Branch:** `main`
   - **Main file path:** `app.py`
6. Click **"Deploy!"**

### Step 5: Wait for Deployment

- Streamlit will build and deploy your app (takes 2-5 minutes)
- Once complete, you'll get a URL like: `https://your-app-name.streamlit.app`
- **That's it! Your app is live!** 🎉

---

## Option 2: Run Locally (For Testing)

If you want to test on your own computer first:

### Step 1: Install Python

1. Go to [python.org/downloads](https://python.org/downloads)
2. Download Python 3.10 or newer
3. Run the installer
4. **IMPORTANT:** Check the box that says "Add Python to PATH"
5. Click "Install Now"

### Step 2: Download the App Files

1. Download all the files from this folder to a location on your computer
2. Create a new folder called `prop-firm-simulator`
3. Put all the files in that folder

### Step 3: Open Command Prompt/Terminal

**On Windows:**
1. Press `Windows key + R`
2. Type `cmd` and press Enter
3. Type: `cd path\to\prop-firm-simulator` (replace with your actual folder path)

**On Mac:**
1. Open Spotlight (Cmd + Space)
2. Type "Terminal" and press Enter
3. Type: `cd path/to/prop-firm-simulator` (replace with your actual folder path)

### Step 4: Install Requirements

In the terminal, type:
```
pip install -r requirements.txt
```
Press Enter and wait for installation to complete.

### Step 5: Run the App

In the terminal, type:
```
streamlit run app.py
```

Your browser should automatically open to `http://localhost:8501` with the app running!

---

## 🔧 Customization Guide

### Adding New Prop Firms

Edit `prop_firms.json` and add a new entry following this format:

```json
"new_firm_50k": {
  "display_name": "New Firm Name - $50K",
  "account_size": 50000,
  "profit_target": 3000,
  "max_trailing_drawdown": 2500,
  "trailing_drawdown_type": "end_of_day",
  "daily_loss_limit": null,
  "min_trading_days": 5,
  "max_trading_days": null,
  "consistency_rule": false,
  "notes": "Any special notes about this firm"
}
```

**Key fields explained:**
- `trailing_drawdown_type`: Either `"end_of_day"` or `"real_time"`
- `daily_loss_limit`: Set to `null` if none, or a number like `1000`
- `consistency_rule`: `true` if firm has a max % from single day rule
- `consistency_max_day_percent`: Only needed if `consistency_rule` is `true`

### Changing the Number of Simulations

In `app.py`, find this line and change the values:
```python
num_sims = st.slider(
    "Number of simulations:",
    min_value=50,
    max_value=500,
    value=200,  # Change this default
    ...
)
```

---

## 📝 Files Included

| File | Purpose |
|------|---------|
| `app.py` | Main application interface |
| `simulation.py` | Monte Carlo simulation engine |
| `prop_firms.json` | Database of prop firm rules |
| `requirements.txt` | Python package dependencies |
| `sample_trades.csv` | Example CSV format for testing |
| `README.md` | This documentation |

---

## ⚠️ Important Disclaimer

This simulation tool is for **educational and planning purposes only**. 

- Results are based on random resampling of historical backtest data
- Past performance does NOT guarantee future results
- Actual trading involves factors not captured in simulations
- Always trade responsibly and only risk capital you can afford to lose

---

## 🆘 Troubleshooting

**"Module not found" error:**
- Make sure you ran `pip install -r requirements.txt`

**"Could not find P&L column" error:**
- Rename your P&L column to exactly `PnL` or `Profit`

**App won't load on Streamlit Cloud:**
- Check that all 5 files are uploaded to your GitHub repo
- Make sure `requirements.txt` is in the root folder

**Charts not displaying:**
- Try refreshing the page
- Make sure you have at least 5 days of trade data

---

## 📞 Need Help?

If you need modifications or have questions, you can:
1. Ask Claude (AI) for help with specific changes
2. Post on the Streamlit community forums
3. Check GitHub Issues for common problems

---

Built with ❤️ for algorithmic traders
