x = "X"
o = "O"
EMPTY = " "
TIE = "TIE"
Num_Squares = 9

def display_instructions():
    print("""
  Welcome to the greatest intellectual challenge of all time: Tic-Tac-Toe.
  This will be a showdown between your human brain and my silicon processor.
  
   You will make your move known by entering a number, 0 - 8. The number
   will correspond to the board position as illustrated:

                    0 | 1 | 2
                    ---------
                    3 | 4 | 5
                    ---------
                    6 | 7 | 8

    Prepare yourself, human. The ultimate battle is about to begin. \n

          """)

def ask_yes_no(question):
    response = None
    while response not in ("y","n"):
        response = input(question).lower()

    return response

def ask_number(question,low,high,step = 1):
    response = None
    while response not in range(low,high,step):
        response = int(input(question))

    return response

def pieces():
    go_first = ask_yes_no("Do you want to go first? (y/n) ")
    if go_first == "y":
        print("\nThen take the first move.   You will need it.")
        human = x
        computer = o
    else:
        print("Your bravery will do you nothing....I will go first. ")
        human = o
        computer = x
        
    return human, computer

def new_board():

    board = []
    for square in range(Num_Squares):
        board.append(EMPTY)
    return board

def display_board(board):
    print("\n\t", board[0], "|", board[1], "|", board[2])
    print("\t", "---------")
    print("\t", board[3], "|", board[4], "|", board[5])
    print("\t", "---------")
    print("\t", board[6], "|", board[7], "|", board[8], "\n")

def legal_moves(board):
    moves = []
    for square in range(Num_Squares):
        if board[square] == EMPTY:
            moves.append(square)
    return moves

def winner(board):
    ways_to_win = [(0,1,2),
                   (3,4,5),
                   (6,7,8),
                   (0,3,6),
                   (1,4,7),
                   (2,5,8),
                   (0,4,8),
                   (2,4,6)]
    for row in ways_to_win:
        if board[row[0]] == board[row[1]] == board[row[2]] != EMPTY:
            winner = board[row[0]]
            return winner
        elif EMPTY not in board:
            return TIE
        else:
            return None

def human_move(board,human):
    legal = legal_moves(board)
    move = None
    while move not in legal:
        move = ask_number("Where Will you move? (0-8): ",0,Num_Squares)
        if move not in legal:
            print("\nThat square is already occupied, foolish human. Choose another.\n")
    
    print("Fine...")
    return move

def computer_move(board,computer,human):
    board = board[:]
    BEST_MOVES = (4,0,2,6,8,1,3,5,7)
    print("I shall take the square number...",end =" ")
    for move in legal_moves(board):
        board[move] = computer
        if winner(board) == computer:
            print(move)
            return move
        board[move] = EMPTY
    for move in legal_moves(board):
        board[move] = human
        if winner(board) == human:
            print(move)
            return move
        board[move] = EMPTY
    for move in BEST_MOVES:
        if move in legal_moves(board):
            print(move)
            return move

def next_turn(turn):
    if turn == x:
        return o
    else:
        return x

def congrat_winner(the_winner,computer,human):
    if the_winner != TIE:
        print(the_winner,"WON!!")
    else:
        print("It's a tie.")

    if the_winner == computer:
        print("As I predicted, human, I am triumphant once more. \n" \
              "Proof that computers are superior to humans in all regards.")
    elif the_winner == human:
        print("No, no! It cannot be! Somehow you tricked me, human. \n" \
        "But never again! I, the computer, so swear it!")
    elif the_winner == TIE:
        print("You were most lucky, human, and somehow managed to tie me. \n" \
        "Celebrate today... for this is the best you will ever achieve.")    
    


def main():
    display_instructions()
    human, computer = pieces()
    turn = x
    board = new_board()
    display_board(board)

    while not winner(board):
        if turn == human:
            move = human_move(board,human)
            board[move] = human
        else:
            move = computer_move(board, computer, human)
            board[move] = computer
        display_board(board)
        turn = next_turn(turn)



    the_winner = winner(board)
    congrat_winner(the_winner,computer,human)

main()
input("\n\nPress the enter key to quit.")
        
    
    
            
    

    
    
    
    
