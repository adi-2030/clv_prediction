import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import joblib

# Load model and top features
model = joblib.load('randomforest_churn_model.pkl')
top_features = joblib.load('top_features.pkl')

# Function to predict churn for single customer
def predict_single():
    try:
        input_data = []
        for feature in top_features:
            val = float(entries[feature].get())
            input_data.append(val)
        df = pd.DataFrame([input_data], columns=top_features)
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[:,1][0]

        if prob > 0.8:
            risk = "High Risk ⚠️"
        elif prob > 0.5:
            risk = "Medium Risk ⚠️"
        else:
            risk = "Low Risk ✅"

        msg = f"Prediction: {risk}\nProbability: {prob*100:.2f}%"
        messagebox.showinfo("Churn Prediction Result", msg)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to predict batch CSV
def predict_batch():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
    if file_path:
        df_new = pd.read_csv(file_path)
        for col in top_features:
            if col not in df_new.columns:
                df_new[col] = 0
        df_new = df_new[top_features]
        df_new['Churn_Prediction'] = model.predict(df_new)
        df_new['Churn_Probability'] = model.predict_proba(df_new)[:,1]
        save_path = filedialog.asksaveasfilename(defaultextension=".csv")
        if save_path:
            df_new.to_csv(save_path, index=False)
            messagebox.showinfo("Batch Prediction", f"Predictions saved to {save_path}")

# Tkinter GUI Setup
root = tk.Tk()
root.title("Telecom Churn Prediction")

canvas = tk.Canvas(root, width=600, height=600)
canvas.pack()

frame = tk.Frame(root)
frame.place(relx=0.5, rely=0.05, anchor='n')

entries = {}
row = 0
for feature in top_features[:10]:  # Show top 10 features for simplicity
    tk.Label(frame, text=feature).grid(row=row, column=0, pady=5)
    entry = tk.Entry(frame)
    entry.grid(row=row, column=1, pady=5)
    entries[feature] = entry
    row += 1

tk.Button(root, text="Predict Single Customer", command=predict_single).place(relx=0.3, rely=0.85)
tk.Button(root, text="Predict Batch CSV", command=predict_batch).place(relx=0.6, rely=0.85)

root.mainloop()

import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------
# Load Model and Features
# -----------------------
model = joblib.load('randomforest_churn_model.pkl')
top_features = joblib.load('top_features.pkl')  # top 30 features

# -----------------------
# Recommendations based on risk
# -----------------------
def churn_recommendation(prob):
    if prob > 0.8:
        return [
            "⚠️ Offer discount to retain customer",
            "⚠️ Assign priority customer support",
            "⚠️ Check if contract plan can be upgraded"
        ]
    elif prob > 0.5:
        return [
            "⚠️ Monitor customer usage",
            "⚠️ Send engagement emails or offers"
        ]
    else:
        return ["✅ Customer seems low-risk"]

# -----------------------
# Single Customer Prediction
# -----------------------
def predict_single():
    try:
        input_data = []
        for feature in top_features:
            val = float(entries[feature].get())
            input_data.append(val)
        df = pd.DataFrame([input_data], columns=top_features)
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[:,1][0]

        # Risk Label
        if prob > 0.8:
            risk = "High Risk ⚠️"
        elif prob > 0.5:
            risk = "Medium Risk ⚠️"
        else:
            risk = "Low Risk ✅"

        msg = f"Prediction: {risk}\nProbability: {prob*100:.2f}%\n\nRecommendations:\n"
        for rec in churn_recommendation(prob):
            msg += f"- {rec}\n"

        messagebox.showinfo("Churn Prediction Result", msg)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# -----------------------
# Batch CSV Prediction
# -----------------------
def predict_batch():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
    if file_path:
        df_new = pd.read_csv(file_path)
        for col in top_features:
            if col not in df_new.columns:
                df_new[col] = 0
        df_new = df_new[top_features]

        batch_pred = model.predict(df_new)
        batch_prob = model.predict_proba(df_new)[:,1]

        df_new['Churn_Prediction'] = ["Churn" if p==1 else "No Churn" for p in batch_pred]
        df_new['Churn_Probability'] = batch_prob
        df_new['Risk_Level'] = ["High Risk ⚠️" if p>0.8 else "Medium Risk ⚠️" if p>0.5 else "Low Risk ✅" for p in batch_prob]
        df_new['Recommendations'] = [churn_recommendation(p) for p in batch_prob]

        save_path = filedialog.asksaveasfilename(defaultextension=".csv")
        if save_path:
            df_new.to_csv(save_path, index=False)
            messagebox.showinfo("Batch Prediction", f"Predictions saved to {save_path}")

# -----------------------
# Feature Importance Plot
# -----------------------
def show_feature_importance():
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': top_features, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10,6))
    plt.barh(feat_imp_df['Feature'][:10], feat_imp_df['Importance'][:10], color='skyblue')
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importance")
    plt.xlabel("Importance")
    plt.show()

# -----------------------
# Tkinter GUI Setup
# -----------------------
root = tk.Tk()
root.title("Telecom Churn Prediction")
root.geometry("700x800")

# Instructions
tk.Label(root, text="📞 Telecom Churn Prediction", font=("Helvetica", 16, "bold")).pack(pady=10)
tk.Label(root, text="Enter customer details below (Top 30 features) and click Predict.", wraplength=600).pack(pady=5)

# Frame for input fields
frame = tk.Frame(root)
frame.pack(pady=10)

canvas = tk.Canvas(frame)
scroll_y = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)

input_frame = tk.Frame(canvas)
canvas.create_window((0,0), window=input_frame, anchor='nw')

entries = {}
row = 0
for feature in top_features:
    tk.Label(input_frame, text=feature).grid(row=row, column=0, pady=2, sticky='w')
    entry = tk.Entry(input_frame, width=20)
    entry.grid(row=row, column=1, pady=2)
    entries[feature] = entry
    row += 1

input_frame.update_idletasks()
canvas.configure(scrollregion=canvas.bbox('all'), yscrollcommand=scroll_y.set, width=650, height=400)
canvas.pack(side="left")
scroll_y.pack(side="right", fill='y')

# Buttons
tk.Button(root, text="Predict Single Customer", bg="lightblue", command=predict_single).pack(pady=10)
tk.Button(root, text="Predict Batch CSV", bg="lightgreen", command=predict_batch).pack(pady=10)
tk.Button(root, text="Show Feature Importance", bg="orange", command=show_feature_importance).pack(pady=10)

root.mainloop()

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from threading import Thread
import time

# -----------------------
# Load model and features
# -----------------------
model = joblib.load('randomforest_churn_model.pkl')
top_features = joblib.load('top_features.pkl')

# -----------------------
# Recommendations
# -----------------------
def churn_recommendation(prob):
    if prob > 0.8:
        return ["⚠️ Offer discount", "⚠️ Priority support", "⚠️ Upgrade plan"]
    elif prob > 0.5:
        return ["⚠️ Monitor usage", "⚠️ Send engagement offers"]
    else:
        return ["✅ Low risk customer"]

# -----------------------
# Predict Single Customer
# -----------------------
def predict_single(progress_label):
    try:
        progress_label.config(text="Predicting...")
        root.update_idletasks()
        # Animation simulation
        for i in range(5):
            progress_label.config(text=f"Predicting{'.'*i}")
            time.sleep(0.2)
            root.update_idletasks()

        input_data = [float(entries[feature].get()) for feature in top_features]
        df = pd.DataFrame([input_data], columns=top_features)
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[:,1][0]

        # Risk label with colors
        if prob > 0.8:
            risk = "High Risk ⚠️"
            color = "red"
        elif prob > 0.5:
            risk = "Medium Risk ⚠️"
            color = "orange"
        else:
            risk = "Low Risk ✅"
            color = "green"

        result_text = f"Prediction: {risk}\nProbability: {prob*100:.2f}%\n\nRecommendations:\n"
        for rec in churn_recommendation(prob):
            result_text += f"- {rec}\n"

        result_label.config(text=result_text, fg=color)
        progress_label.config(text="Done ✅")
    except Exception as e:
        messagebox.showerror("Error", str(e))
        progress_label.config(text="Error ❌")

# -----------------------
# Predict Batch CSV
# -----------------------
def predict_batch():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
    if file_path:
        df_new = pd.read_csv(file_path)
        for col in top_features:
            if col not in df_new.columns:
                df_new[col] = 0
        df_new = df_new[top_features]

        batch_pred = model.predict(df_new)
        batch_prob = model.predict_proba(df_new)[:,1]

        df_new['Churn_Prediction'] = ["Churn" if p==1 else "No Churn" for p in batch_pred]
        df_new['Churn_Probability'] = batch_prob
        df_new['Risk_Level'] = ["High ⚠️" if p>0.8 else "Medium ⚠️" if p>0.5 else "Low ✅" for p in batch_prob]
        df_new['Recommendations'] = [churn_recommendation(p) for p in batch_prob]

        save_path = filedialog.asksaveasfilename(defaultextension=".csv")
        if save_path:
            df_new.to_csv(save_path, index=False)
            messagebox.showinfo("Batch Prediction", f"Predictions saved to {save_path}")

# -----------------------
# Feature Importance
# -----------------------
def show_feature_importance():
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': top_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10,6))
    plt.barh(feat_imp_df['Feature'][:10], feat_imp_df['Importance'][:10], color='skyblue')
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importance")
    plt.show()

# -----------------------
# GUI Setup
# -----------------------
root = tk.Tk()
root.title("📞 Telecom Churn Predictor")
root.geometry("800x700")

# Tabs
tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab_control.add(tab1, text="Single Customer")
tab_control.add(tab2, text="Batch Prediction")
tab_control.add(tab3, text="Feature Importance")
tab_control.pack(expand=1, fill="both")

# -----------------------
# Tab 1: Single Customer
# -----------------------
canvas = tk.Canvas(tab1)
scroll_y = tk.Scrollbar(tab1, orient="vertical", command=canvas.yview)
input_frame = tk.Frame(canvas)
canvas.create_window((0,0), window=input_frame, anchor='nw')
canvas.configure(yscrollcommand=scroll_y.set, width=750, height=400)
canvas.pack(side="left")
scroll_y.pack(side="right", fill='y')

entries = {}
row = 0
for feature in top_features:
    tk.Label(input_frame, text=feature).grid(row=row, column=0, pady=2, sticky='w')
    entry = tk.Entry(input_frame, width=20)
    entry.grid(row=row, column=1, pady=2)
    entries[feature] = entry
    row += 1
input_frame.update_idletasks()
canvas.configure(scrollregion=canvas.bbox('all'))

progress_label = tk.Label(tab1, text="")
progress_label.pack(pady=5)
result_label = tk.Label(tab1, text="", justify='left', font=("Helvetica", 12))
result_label.pack(pady=5)

tk.Button(tab1, text="Predict Single Customer", bg="lightblue",
          command=lambda: Thread(target=predict_single, args=(progress_label,)).start()).pack(pady=10)

# -----------------------
# Tab 2: Batch Prediction
# -----------------------
tk.Button(tab2, text="Predict Batch CSV", bg="lightgreen", command=predict_batch).pack(pady=20)

# -----------------------
# Tab 3: Feature Importance
# -----------------------
tk.Button(tab3, text="Show Feature Importance", bg="orange", command=show_feature_importance).pack(pady=20)

root.mainloop()
