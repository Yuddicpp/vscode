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
import random

class GUI(tk.Tk):
    """
        Draw the GUI of game
    """
    def __init__(self,N,N_Obstacle):
        """
            N x N
        """
        super().__init__()

        self.N = N
        self.HEIGHT = 400
        self.WIDTH = 400
        self.margin = self.HEIGHT//N

        self.canvas = tk.Canvas(self, bg='white', height=self.HEIGHT, width=self.WIDTH)
        self.canvas.pack()
        self.title('Cat and Mouse')

        # Location of the cat, mouse and Obstacles
        self.Cat_loc = np.array([0,0])
        self.Mouse_loc = np.array([N-1,N-1])
        self.Obstacle_loc = np.vstack((random.sample(range(1,N-1),N_Obstacle),random.sample(range(1,N-1),N_Obstacle))).T
        self.Q = np.zeros((self.N,self.N))

        # Get the image of cat and mouse
        self.Cat_image = ImageTk.PhotoImage(Image.open("cat.png").resize((self.margin,self.margin)))
        self.Mouse_image = ImageTk.PhotoImage(Image.open("mouse.jpg").resize((self.margin,self.margin)))
    
    def show(self):
        # Draw the image of cat and Mouse
        self.canvas.create_image(self.Cat_loc[1]*self.margin, self.Cat_loc[0]*self.margin, anchor='nw', image=self.Cat_image)
        self.canvas.create_image(self.Mouse_loc[1]*self.margin, self.Mouse_loc[0]*self.margin, anchor='nw', image=self.Mouse_image)

        # Draw the image of the map
        for i in range(self.N+1):
            self.canvas.create_line(0,self.margin*i,self.WIDTH,self.margin*i)
            self.canvas.create_line(self.margin*i,0,self.margin*i,self.HEIGHT)
        
        # Draw the image of obstacles
        for loc in self.Obstacle_loc:
            self.canvas.create_rectangle(self.margin*loc[1],self.margin*loc[0],self.margin*loc[1]+self.margin,self.margin*loc[0]+self.margin,fill='black')
        
        self.mainloop()


class SARSA(GUI):
    def __init__(self,N,N_Obstacle):
        super().__init__(N,N_Obstacle)



if __name__ == '__main__':
    Pic = SARSA(4,2)
    print(Pic.N)
