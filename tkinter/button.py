
from tkinter import *

window = Tk()
window.title("Welcome to My App")
lbl = Label(window, text="Hello!", font=("Arial", 18))
lbl.grid(column=0, row=0)

btn = Button(window, text="Submit", bg="blue", fg="red")
btn.grid(column=1, row=0)

window.geometry('350x200')
window.mainloop()
