from PIL import ImageGrab
from tkinter import *

root = Tk()
cv = Canvas(root)
cv.create_rectangle(10,10,50,50)
cv.pack()
root.mainloop()

def getter(widget):
    x=root.winfo_rootx()+widget.winfo_x()
    y=root.winfo_rooty()+widget.winfo_y()
    x1=x+widget.winfo_width()
    y1=y+widget.winfo_height()
    ImageGrab.grab().crop((x,y,x1,y1)).save("file path here")