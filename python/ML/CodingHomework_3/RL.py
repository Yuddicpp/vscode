# /*
#  * @Author: Yuddi 
#  * @Date: 2021-12-23 18:24:39 
#  * @Last Modified by:    
#  * @Last Modified time: 2021-12-23 18:24:39 
#  */


import time
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
        self.title('Cat and Mouse')

        # Location of the cat, mouse and Obstacles
        self.Cat_loc = np.array([0,0])
        self.Mouse_loc = np.array([N-1,N-1])
        # self.Obstacle_loc = np.vstack((random.sample(range(1,N-1),N_Obstacle),random.sample(range(1,N-1),N_Obstacle))).T
        self.Obstacle_loc = np.array([[1,2],[2,1]])

        # N * N * 4 
        # 0：Up 1: Right 2: Down 3: Left
        self.Q = np.zeros((self.N,self.N,4))

        self.reward = -np.ones((self.N,self.N))
        self.reward[self.Mouse_loc[0],self.Mouse_loc[1]] = 10
        for loc in self.Obstacle_loc:
            self.reward[loc[0],loc[1]] = -10

        # Get the image of cat and mouse
        self.Cat_image = ImageTk.PhotoImage(Image.open("cat.png").resize((self.margin-1,self.margin-1)))
        self.Mouse_image = ImageTk.PhotoImage(Image.open("mouse.jpg").resize((self.margin-1,self.margin-1)))

        self.canvas.pack()

        
    
    def show(self):
        self.canvas.delete('all')

        # Draw the image of cat and Mouse
        self.canvas.create_image(self.Cat_loc[1]*self.margin+1, self.Cat_loc[0]*self.margin+1, anchor='nw', image=self.Cat_image)
        self.canvas.create_image(self.Mouse_loc[1]*self.margin+1, self.Mouse_loc[0]*self.margin+1, anchor='nw', image=self.Mouse_image)

        # Draw the image of obstacles
        for loc in self.Obstacle_loc:
            self.canvas.create_rectangle(self.margin*loc[1],self.margin*loc[0],self.margin*loc[1]+self.margin,self.margin*loc[0]+self.margin,fill='black')
        
        # Draw the image of the map
        for i in range(self.N):
            self.canvas.create_line(0,self.margin*i,self.WIDTH,self.margin*i)
            self.canvas.create_line(self.margin*i,0,self.margin*i,self.HEIGHT)

        self.update()
        time.sleep(0.01)
        



class SARSA(GUI):
    def __init__(self,N,N_Obstacle,EPSILON,Alpha,Gamma):
        super().__init__(N,N_Obstacle)
        # 0：Up 1: Right 2: Down 3: Left
        self.action = 0
        self.epsilon = EPSILON
        self.alpha = Alpha
        self.gamma = Gamma

    def epsilon_greedy(self,state):
        if random.random() > self.epsilon:
            # print(random.random(),self.epsilon)
            return random.randint(0,3)
        else:
            return np.argmax(self.Q[state[0],state[1],:])
        
    def move(self):
        if(self.action==0):
            s_next = np.array((self.Cat_loc[0]-1,self.Cat_loc[1]))
            if s_next[0] < 0:
                s_next[0] = 0
            r = self.reward[s_next[0],s_next[1]]
        elif(self.action==1):
            s_next = np.array((self.Cat_loc[0],self.Cat_loc[1]+1))
            if s_next[1] > self.N-1:
                s_next[1] = self.N-1
            r = self.reward[s_next[0],s_next[1]]
        elif(self.action==2):
            s_next = np.array((self.Cat_loc[0]+1,self.Cat_loc[1]))
            if s_next[0] > self.N - 1:
                s_next[0] = self.N - 1
            r = self.reward[s_next[0],s_next[1]]
        elif(self.action==3):
            s_next = np.array((self.Cat_loc[0],self.Cat_loc[1]-1))
            if s_next[1] < 0:
                s_next[1] = 0
            r = self.reward[s_next[0],s_next[1]]
        return s_next,r


    def update_Q(self) -> None:
        self.show()
        r_sum = 0
        self.Cat_loc = np.array([0,0])
        self.action = self.epsilon_greedy(self.Cat_loc)
        # print(self.Cat_loc,self.action)
        while(1):
            s_next,r = self.move()
            r_sum += r
            action_ = self.epsilon_greedy(s_next)
            self.Q[self.Cat_loc[0],self.Cat_loc[1],self.action] = self.Q[self.Cat_loc[0],self.Cat_loc[1],self.action] + self.alpha *(r + self.gamma * self.Q[s_next[0],s_next[1],action_]-self.Q[self.Cat_loc[0],self.Cat_loc[1],self.action])
            self.Cat_loc = s_next
            self.action = action_
            if (self.Cat_loc==self.Obstacle_loc).all(1).any():
                # print('over')
                break
            if((self.Cat_loc==self.Mouse_loc).all()):
                print('win')
                break
            # print(self.Cat_loc,self.action)
            self.show()
        print(r_sum)
    



class Q_Learning(GUI):
    def __init__(self,N,N_Obstacle):
        super().__init__(N,N_Obstacle)


if __name__ == '__main__':
    sarsa = SARSA(4,2,1,0.01,0.9)
    for i in range(1000):
        sarsa.update_Q()
    sarsa.mainloop()

