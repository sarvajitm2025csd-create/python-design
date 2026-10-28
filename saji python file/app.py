import tkinter as tk
from tkinter import ttk, messagebox
import random
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Dataset and Models
data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42,
    stratify=data.target
)

FEATURES = list(data.feature_names)

KEY_FEATURES = [
    "worst perimeter",
    "worst area",
    "worst concave points",
    "mean concave points"
]

MODELS = {
    "Logistic Regression": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "KNN": lambda: make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

BG = "#172554"
CARD = "#0F172A"
GOLD = "#FACC15"


class BreastCancerApp:

    def __init__(self, root):
        self.root = root
        self.root.title("BREAST CANCER PREDICTION")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)

        self.model = None
        self.sample = None
        self.sample_label = None
        self.sample_text = {}

        tk.Label(
            root,
            text="BREAST CANCER PREDICTION",
            font=("Arial Black", 20),
            bg=BG,
            fg="white"
        ).pack(pady=(10, 2))

        tk.Label(
            root,
            text="Name: Sarvajit M    |    Reg No: 251701050",
            font=("Arial", 11),
            bg=BG,
            fg=GOLD
        ).pack()

        # Model selection
        top = tk.Frame(root, bg=BG)
        top.pack(pady=10)

        tk.Label(
            top,
            text="Select Model:",
            font=("Arial", 12, "bold"),
            bg=BG,
            fg="white"
        ).pack(side="left", padx=5)

        self.model_var = tk.StringVar(
            value="Logistic Regression"
        )

        ttk.Combobox(
            top,
            textvariable=self.model_var,
            values=list(MODELS),
            state="readonly",
            width=20,
            font=("Arial", 11)
        ).pack(side="left", padx=5)

        tk.Button(
            top,
            text="TRAIN & EVALUATE",
            font=("Arial", 11, "bold"),
            bg="#22C55E",
            fg="black",
            command=self.train_model
        ).pack(side="left", padx=10)

        self.acc_label = tk.Label(
            root,
            text="",
            font=("Arial", 13, "bold"),
            bg=BG,
            fg=GOLD
        )
        self.acc_label.pack()

        # Classification report and confusion matrix
        mid = tk.Frame(root, bg=BG)
        mid.pack(padx=20, pady=10)

        self.report_box = tk.Text(
            mid,
            width=52,
            height=12,
            bg=CARD,
            fg="white",
            font=("Courier New", 10),
            highlightthickness=2,
            highlightbackground="#38BDF8"
        )
        self.report_box.pack(side="left", padx=(0, 15))

        self.canvas = tk.Canvas(
            mid,
            width=300,
            height=230,
            bg=CARD,
            highlightthickness=2,
            highlightbackground="#38BDF8"
        )
        self.canvas.pack(side="left")

        # Patient prediction panel
        panel = tk.LabelFrame(
            root,
            text=" PREDICT FOR A PATIENT ",
            font=("Arial", 11, "bold"),
            bg=BG,
            fg="white"
        )
        panel.pack(padx=20, pady=5, fill="x")

        self.entries = {}

        for i, name in enumerate(KEY_FEATURES):
            tk.Label(
                panel,
                text=name,
                font=("Arial", 10),
                bg=BG,
                fg="white"
            ).grid(
                row=i // 2,
                column=(i % 2) * 2,
                padx=8,
                pady=4,
                sticky="e"
            )

            var = tk.StringVar()

            tk.Entry(
                panel,
                textvariable=var,
                width=12,
                font=("Arial", 11)
            ).grid(
                row=i // 2,
                column=(i % 2) * 2 + 1,
                padx=8
            )

            self.entries[name] = var

        # Buttons
        btns = tk.Frame(root, bg=BG)
        btns.pack(pady=8)

        tk.Button(
            btns,
            text="RANDOM TEST PATIENT",
            font=("Arial", 11, "bold"),
            bg="#38BDF8",
            fg="black",
            command=self.load_sample
        ).pack(side="left", padx=8)

        tk.Button(
            btns,
            text="PREDICT",
            font=("Arial", 11, "bold"),
            bg="#22C55E",
            fg="black",
            command=self.predict
        ).pack(side="left", padx=8)

        tk.Button(
            btns,
            text="EXIT",
            font=("Arial", 11, "bold"),
            bg="#EF4444",
            fg="white",
            command=root.destroy
        ).pack(side="left", padx=8)

        self.result_label = tk.Label(
            root,
            text="Result: -",
            font=("Arial Black", 16),
            bg=BG,
            fg="white"
        )
        self.result_label.pack(pady=(4, 2))

        self.actual_label = tk.Label(
            root,
            text="",
            font=("Arial", 11),
            bg=BG,
            fg="white"
        )
        self.actual_label.pack(pady=(0, 10))

        self.train_model()

    def train_model(self):
        name = self.model_var.get()
        self.model = MODELS[name]()

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        cv = cross_val_score(
            MODELS[name](),
            data.data,
            data.target,
            cv=5
        ).mean()

        report = classification_report(
            y_test,
            y_pred,
            target_names=data.target_names
        )

        self.acc_label.config(
            text=(
                f"Test Accuracy: {acc * 100:.2f}%"
                f"   |   5-Fold CV: {cv * 100:.2f}%"
            )
        )

        self.report_box.delete("1.0", "end")
        self.report_box.insert(
            "end",
            f"Model: {name}\n\n{report}"
        )

        self.draw_matrix(confusion_matrix(y_test, y_pred))

    def draw_matrix(self, cm):
        c = self.canvas
        c.delete("all")

        names = ["malignant", "benign"]

        c.create_text(
            150, 14,
            text="CONFUSION MATRIX",
            fill="white",
            font=("Arial", 11, "bold")
        )

        c.create_text(
            195, 36,
            text="Predicted",
            fill=GOLD,
            font=("Arial", 10)
        )

        c.create_text(
            40, 62,
            text="Actual",
            fill=GOLD,
            font=("Arial", 10)
        )

        top = max(cm.max(), 1)

        for j in range(2):
            c.create_text(
                150 + j * 90,
                56,
                text=names[j],
                fill="white",
                font=("Arial", 9)
            )

        for i in range(2):
            c.create_text(
                98,
                110 + i * 80,
                text=names[i],
                fill="white",
                anchor="e",
                font=("Arial", 9)
            )

            for j in range(2):
                r = cm[i][j] / top

                color = "#%02x%02x%02x" % (
                    int(235 - 190 * r),
                    int(240 - 150 * r),
                    255 - int(40 * r)
                )

                x = 105 + j * 90
                y = 70 + i * 80

                c.create_rectangle(
                    x, y,
                    x + 90, y + 80,
                    fill=color,
                    outline="#0F172A",
                    width=2
                )

                c.create_text(
                    x + 45,
                    y + 40,
                    text=str(cm[i][j]),
                    fill="white" if r > 0.5 else "black",
                    font=("Arial Black", 18)
                )

    def load_sample(self):
        idx = random.randrange(len(X_test))

        self.sample = X_test[idx]
        self.sample_label = y_test[idx]

        for name, var in self.entries.items():
            text = f"{self.sample[FEATURES.index(name)]:.4f}"
            var.set(text)
            self.sample_text[name] = text

        self.result_label.config(
            text="Result: -",
            fg="white"
        )

        self.actual_label.config(
            text="Sample loaded. Click PREDICT."
        )

    def predict(self):
        edited = any(
            self.entries[n].get() != self.sample_text.get(n)
            for n in KEY_FEATURES
        )

        if self.sample is not None and not edited:
            vec = self.sample.copy()
        else:
            vec = X_train.mean(axis=0)

            try:
                for name, var in self.entries.items():
                    vec[FEATURES.index(name)] = float(var.get())
            except ValueError:
                messagebox.showerror(
                    "Invalid Input",
                    "Please enter numeric values in all fields."
                )
                return

        pred = self.model.predict([vec])[0]
        prob = self.model.predict_proba([vec])[0][pred]

        label = data.target_names[pred].upper()

        self.result_label.config(
            text=f"Result: {label} ({prob * 100:.1f}% confidence)",
            fg="#EF4444" if pred == 0 else "#22C55E"
        )

        if self.sample is not None and not edited:
            self.actual_label.config(
                text=(
                    "Actual diagnosis: "
                    f"{data.target_names[self.sample_label].upper()}"
                )
            )
        else:
            self.actual_label.config(
                text="(Other features set to dataset average)"
            )


# Main Application
root = tk.Tk()
app = BreastCancerApp(root)
root.mainloop()