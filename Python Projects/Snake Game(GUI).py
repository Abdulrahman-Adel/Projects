import turtle as t
import random as rd

t.bgcolor('blue')

Snake = t.Turtle()
Snake.shape('square')
Snake.speed(0)
Snake.penup()
Snake.hideturtle()

leaf = t.Turtle()
leaf_shape = ((0,0),(14,2),(18,6),(20,20),(6,18),(2,14))
t.register_shape('leaf',leaf_shape)
leaf.shape('leaf')
leaf.color('red')
leaf.penup()
leaf.hideturtle()
leaf.speed()

game_started = False
text_turtle = t.Turtle()
text_turtle.write('Press SPACE to start',align='center',font=('Arial',16,'bold'))
text_turtle.hideturtle()

score_turtle = t.Turtle()
score_turtle.hideturtle()
score_turtle.speed(0)

def outside_window():
    left_wall = -t.window_width()/2
    right_wall = t.window_width()/2
    top_wall = t.window_height()/2
    bottom_wall = -t.window_height()/2
    (x,y) = Snake.pos()
    outside = x < left_wall or  x > right_wall or  y < bottom_wall or y > top_wall
    return outside

def game_over():
    Snake.color('yellow')
    leaf.color('yellow')
    t.penup()
    t.hideturtle()
    t.write('GAME OVER!',align='center' , font=('Aerial',30,'normal'))

def display_score(current_score):
    score_turtle.clear()
    score_turtle.penup()
    x = (t.window_width() / 2)-50
    y = (t.window_height() / 2)-50
    score_turtle.setpos(x,y)
    score_turtle.write(str(current_score) , align = 'right',font=('Arial',40,'bold'))

def place_leaf():
    leaf.hideturtle()
    leaf.setx(rd.randint(-200,200))
    leaf.sety(rd.randint(-200,200))
    leaf.showturtle()

def start_game():
    global game_started
    if game_started:
        return
    game_started = True

    score = 0
    text_turtle.clear()

    Snake_speed = 2
    Snake_length = 3
    Snake.shapesize(1,Snake_length,1)
    Snake.showturtle()
    display_score(score)
    place_leaf()

    while True:
        Snake.forward(Snake_speed)
        if Snake.distance(leaf)<20:
            place_leaf()
            Snake_length = Snake_length + 1
            Snake.shapesize(1,Snake_length,1)
            Snake_speed = Snake_speed + 1
            score = score + 10
            display_score(score)
        if outside_window():
            game_over()
            break

def move_up():
    if Snake.heading() == 0 or Snake.heading() == 180:
        Snake.setheading(90)

def move_down():
    if Snake.heading() == 0 or Snake.heading() == 180:
        Snake.setheading(270)

def move_left():
    if Snake.heading() == 90 or Snake.heading() == 270:
        Snake.setheading(180)

def move_right():
    if Snake.heading() == 90 or Snake.heading() == 270:
        Snake.setheading(0)

t.onkey(start_game,'space')
t.onkey(move_up,'Up')
t.onkey(move_right,'Right')
t.onkey(move_down,'Down')
t.onkey(move_left,'Left')
t.listen()
t.mainloop()
