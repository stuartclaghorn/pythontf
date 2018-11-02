from tkinter import *

window = Tk()
window.title("Welcome to My App")
lbl = Label(window, text="Hello!", font=("Arial Bold", 50))
lbl.grid(column=0, row=0)

window.geometry('350x200')
window.mainloop()
