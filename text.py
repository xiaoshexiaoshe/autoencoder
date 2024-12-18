import tkinter as tk
from tkinter import *
from tkinter import ttk
import sqlite3

class app_App(tk.Frame):
    def __init__(self,master=None):
        super().__init__(master)
        self.master=master
        self.pack
        self.create_widgets()
        self.create_datebase()

    def create_widgets(self):

        self.name_label = tk.Label(root,text="请输入姓名:")
        self.name_label.pack()

        self.name_entry = tk.Entry(root,width=10)
        self.name_entry.pack()

        subjects = ["数学", "语文", "英语", "体育", "化学", "生物", "物理"]
        self.subject = tk.Label(root, text="选择科目: ")
        self.subject.pack()

        self.subject_spinbox = ttk.Spinbox(root, values=subjects, width=10)
        self.subject_spinbox.pack()

        self.score_label = tk.Label(root, text="输入成绩:")
        self.score_label.pack()

        self.score_spinbox= tk.Spinbox(root,from_=0,to=100,width =10)
        self.score_spinbox.pack()

        self.submit_button=tk.Button(root,text="提交",command=self.show_del)
        self.submit_button.pack()

        self.result_label = tk.Label(root,text="",font=("Arial",11))
        self.result_label.pack()


    def create_datebase(self):
        self.connection = sqlite3.connect('chengji.db')
        self.cursor = self.connection.cursor()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS grades(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                subject TEXT NOT NULL,
                score TEXT NOT NULL
            )
        ''')
        self.connection.commit()


        def submit_date(self):
            name = self.name_entry.get()
            subject = self.subject_spinbox.get()
            score = self.score_spinbox.get()

            self.cursor.excute('''
                INSERT INTO score (name, subject, score) VALUES (? ? ?)
            ''',(name,subject,score))
            self.connection.commit



    def show_del(self):
        name = self.name_entry.get()
        subject = self.subject_spinbox.get()
        score = self.score_spinbox.get()
        result_text=f"姓名: {name},科目: {subject},成绩: {score}"
        self.result_label.config(text=result_text)

    def __del__(self):
        self.connection.close()


if __name__ =='__main__':
    root = tk.Tk()
    root.geometry("500x300+1000+1000")
    root.title("请选择科目和成绩")
    app = app_App(master=root)
    root.mainloop()

