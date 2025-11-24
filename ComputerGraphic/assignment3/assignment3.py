import turtle
import math

class graphic:

    def draw_line(self):
        turtle.setup(800, 800, 0, 0)
        pen=turtle.Turtle()
        pen.hideturtle()
        turtle.hideturtle()

        pen.pendown()
        turtle.goto((100,100))
        turtle.done()

    def draw_circle(self):
        turtle.setup(800,800,0,0)
        pen=turtle.Turtle()
        pen.hideturtle()
        turtle.hideturtle()
        
        pen.pendown()
        turtle.circle(50)

        turtle.done()

    def draw_arc(self):
        turtle.setup(800,800,0,0)
        pen=turtle.Turtle()
        pen.hideturtle()
        turtle.hideturtle()

        pen.pendown()
        turtle.circle(100, 100)
        turtle.done()

    def draw_ellipse(self, x, y, width, height, color='black'):
        turtle.setup(800,800,0,0)
        turtle.penup()
        turtle.goto(x+width/2, y)
        turtle.pendown()
        turtle.color(color)
        turtle.begin_fill()

        for i in range(360):
            angle=i*3.14159/180
            dx=width/2*math.cos(angle)
            dy=height/2*math.sin(angle)
            turtle.goto(x+dx, y+dy)
        turtle.end_fill()
        turtle.hideturtle()
        turtle.done()

    def draw_rectangle(self, width=60, height=100):
        turtle.setup(800, 800, 0, 0)
        pen = turtle.Turtle()
        pen.hideturtle()
        pen.penup()
  
        start_x = -width / 2
        start_y = height / 2
        pen.goto(start_x, start_y) 
        pen.pendown()
        
        pen.forward(width)         
        pen.right(90)              
        pen.forward(height)    
        pen.right(90)             
        pen.forward(width)       
        pen.right(90)            
        pen.forward(height)        
        pen.right(90)            
        
        turtle.done()




graphic=graphic()
graphic.draw_rectangle(200, 100)