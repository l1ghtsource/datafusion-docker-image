import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os
import numpy as np


class CSVMarkupApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Markup Tool")
        self.root.geometry("700x450")

        self.df = None
        self.current_index = 0
        self.filename = None
        self.is_key_needed_var = tk.BooleanVar()
        self.is_value_needed_var = tk.BooleanVar()

        self.create_widgets()
        
    def create_widgets(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.load_button = tk.Button(top_frame, text="Загрузить CSV", command=self.load_csv)
        self.load_button.pack(side=tk.LEFT)
        
        self.filename_label = tk.Label(top_frame, text="Файл не выбран")
        self.filename_label.pack(side=tk.LEFT, padx=10)

        self.data_frame = tk.Frame(self.root)
        self.data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(self.data_frame, text="Key:").grid(row=0, column=0, sticky=tk.W)
        self.key_display = tk.Text(self.data_frame, height=3, width=60)
        self.key_display.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(self.data_frame, text="Typical Values:").grid(row=1, column=0, sticky=tk.W)
        self.values_display = tk.Text(self.data_frame, height=10, width=60)
        self.values_display.grid(row=1, column=1, padx=5, pady=5)

        self.checkbox_frame = tk.Frame(self.data_frame)
        self.checkbox_frame.grid(row=2, column=1, sticky=tk.W, pady=10)

        checkbox_style = {'font': ('Arial', 12), 'padx': 5, 'pady': 5, 'anchor': 'w', 'width': 15, 'height': 2}

        self.key_checkbox = tk.Checkbutton(self.checkbox_frame, text="Is Key Needed", 
                                        variable=self.is_key_needed_var, 
                                        command=self.toggle_key_needed,
                                        **checkbox_style)
        self.key_checkbox.pack(side=tk.LEFT, padx=20)

        self.value_checkbox = tk.Checkbutton(self.checkbox_frame, text="Is Value Needed", 
                                        variable=self.is_value_needed_var, 
                                        command=self.toggle_value_needed,
                                        **checkbox_style)
        self.value_checkbox.pack(side=tk.RIGHT, padx=20)

        nav_frame = tk.Frame(self.root)
        nav_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.prev_button = tk.Button(nav_frame, text="←", command=self.prev_row, width=10, height=2)
        self.prev_button.pack(side=tk.LEFT)
        
        self.counter_label = tk.Label(nav_frame, text="0/0", font=('Arial', 12))
        self.counter_label.pack(side=tk.LEFT, expand=True)
        
        self.next_button = tk.Button(nav_frame, text="→", command=self.next_row, width=10, height=2)
        self.next_button.pack(side=tk.RIGHT)

        self.disable_controls()
        
    def disable_controls(self):
        self.prev_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)
        self.key_checkbox.config(state=tk.DISABLED)
        self.value_checkbox.config(state=tk.DISABLED)
        
    def enable_controls(self):
        self.prev_button.config(state=tk.NORMAL)
        self.next_button.config(state=tk.NORMAL)
        self.key_checkbox.config(state=tk.NORMAL)
        self.value_checkbox.config(state=tk.NORMAL)
        
    def load_csv(self):
        """Загрузка CSV файла"""
        self.filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Выберите CSV файл",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        
        if not self.filename:
            return
            
        try:
            self.df = pd.read_csv(self.filename)

            required_columns = ["key", "typical_values"]
            if not all(col in self.df.columns for col in required_columns):
                messagebox.showerror("Ошибка", "CSV файл должен содержать столбцы 'key' и 'typical_values'")
                return

            if "is_key_needed" not in self.df.columns:
                self.df["is_key_needed"] = np.nan
                
            if "is_value_needed" not in self.df.columns:
                self.df["is_value_needed"] = np.nan

            self.filename_label.config(text=os.path.basename(self.filename))
            self.current_index = 0
            self.update_display()
            self.enable_controls()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")
    
    def update_display(self):
        """Обновляет отображение текущей строки данных"""
        if self.df is None or self.df.empty:
            return

        self.key_display.delete(1.0, tk.END)
        self.values_display.delete(1.0, tk.END)

        row = self.df.iloc[self.current_index]
        self.key_display.insert(tk.END, str(row["key"]))
        self.values_display.insert(tk.END, str(row["typical_values"]))

        key_value = row["is_key_needed"]
        if pd.isna(key_value):
            self.is_key_needed_var.set(False)
        else:
            self.is_key_needed_var.set(bool(key_value))

        value_needed = row["is_value_needed"]
        if pd.isna(value_needed):
            self.is_value_needed_var.set(False)
        else:
            self.is_value_needed_var.set(bool(value_needed))

        self.counter_label.config(text=f"{self.current_index + 1}/{len(self.df)}")

        self.prev_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_index < len(self.df) - 1 else tk.DISABLED)
    
    def next_row(self):
        """Переход к следующей строке"""
        if self.df is None or self.current_index >= len(self.df) - 1:
            return
        
        self.current_index += 1
        self.update_display()
    
    def prev_row(self):
        """Переход к предыдущей строке"""
        if self.df is None or self.current_index <= 0:
            return
        
        self.current_index -= 1
        self.update_display()
    
    def toggle_key_needed(self):
        """Обновляет значение is_key_needed и сохраняет CSV"""
        if self.df is None:
            return

        new_value = self.is_key_needed_var.get()

        self.df.at[self.current_index, "is_key_needed"] = new_value

        try:
            self.df.to_csv(self.filename, index=False)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")
    
    def toggle_value_needed(self):
        """Обновляет значение is_value_needed и сохраняет CSV"""
        if self.df is None:
            return

        new_value = self.is_value_needed_var.get()

        self.df.at[self.current_index, "is_value_needed"] = new_value

        if new_value == True:
            self.is_key_needed_var.set(True)
            self.df.at[self.current_index, "is_key_needed"] = True
        
        try:
            self.df.to_csv(self.filename, index=False)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVMarkupApp(root)
    root.mainloop()