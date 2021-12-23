# /*
#  * @Author: Yuddi 
#  * @Date: 2021-12-23 18:24:39 
#  * @Last Modified by:    
#  * @Last Modified time: 2021-12-23 18:24:39 
#  */


import tkinter as tk
from PIL import Image
from PIL import ImageTk
import numpy as np

class GUI(tk.Tk):
    """
        Draw the GUI of game
    """
    def __init__(self,N):
        """
            N x N
        """
        super().__init__()

        HEIGHT = 400
        WIDTH = 400
        margin = HEIGHT//N

        self.canvas = tk.Canvas(self, bg='white', height=HEIGHT, width=WIDTH)
        self.title('Cat and Mouse')

        self.Cat_loc = np.array([0,0])
        self.Mouse_loc = np.array([N-1,N-1])

        self.Cat_image = ImageTk.PhotoImage(Image.open("cat.png").resize((margin,margin)))
        self.Mouse_image = ImageTk.PhotoImage(Image.open("mouse.jpg").resize((margin,margin)))
        
        self.canvas.create_image(self.Cat_loc[0]*margin, self.Cat_loc[1]*margin, anchor='nw', image=self.Cat_image)
        self.canvas.create_image(self.Mouse_loc[0]*margin, self.Mouse_loc[1]*margin, anchor='nw', image=self.Mouse_image)

        for i in range(N+1):
            self.canvas.create_line(0,margin*i,WIDTH,margin*i)
            self.canvas.create_line(margin*i,0,margin*i,HEIGHT)
        self.canvas.pack()
        self.mainloop()


if __name__ == '__main__':
    Pic = GUI(4)
