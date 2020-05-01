# -*- coding: utf-8 -*-


from enum import Enum

class MapSite():
    def Enter(self):
        raise NotImplementedError("Abstract Base Class Method")
        
        
class Direction(Enum):
    North = 0
    East = 1
    South = 2
    West = 3

class Room(MapSite):
    def __init__(self,room_no):
        self._sides = [MapSite] * 4
        self._roomNumber = int(room_no)
        
    def GetSide(self, Direction):
        return self._sides[Direction]
   
    def SetSide(self,Direction,MapSite):
        self._sides[Direction] = MapSite
        
    def Enter(self):
        print("   You Have Enter Room: ",str(self._roomNumber))
        
class Wall(MapSite):
    def Enter(self):
        print("   *You Just Ran Into A Wall...")

class Door(MapSite):
    def __init__(self,Room1=None,Room2=None):
        self._room1 = Room1
        self._room2 = Room2
        self._isOpen = False
    
    def OtherSideFrom(self,Room):
        print("\tDoor obj: This door is a side of Room: {}".format(Room._roomNumber))
        if 1 == Room._roomNumber:
            other_room = self._room2
        else:
            other_room = self._room1
        return other_room
    def Enter(self):
        if self._isOpen: print("   ****You Have Passed Through This Door...")
        else: print("  ****This door needs tO be opened before you can pass through it....")
        
class Maze():
    def __init__(self):
        self._rooms = {}
        
    def AddRoom(self,room):
        self._rooms[room._roomNumber] = room
    
    def RoomNo(self,room_number):
        return self._rooms[room_number]


#Abstract Factory design pattern
class MazeFactory():
    @classmethod
    def MakeMaze(cls):
        return Maze()
    
    @classmethod
    def MakeWall(cls):
        return Wall()
    
    @classmethod
    def MakeRoom(cls , n):
        return Room(n)
    
    @classmethod
    def MakeDoor(cls,r1,r2):
        return Door(r1,r2)
    
class MazeBuilder():
     def __init__(self):
         pass
     
     def BuildMaze(self):
         pass
         
     def BuildRoom(self,room):
        pass
    
     def BuildDoor(self,roomFrom,roomTo):
        pass
     
     def GetMaze(self):
        return None
    
    
class EnchantedMazeFactory(MazeFactory):
      @classmethod
      def MakeRoom(cls,n):
          return EnchantedRoom(n,cls.CastSpell())
     
      @classmethod
      def MakeDoor(cls,r1,r2):
          return DoorNeedingSpell(r1,r2)
      
      @classmethod
      def CastSpell(cls):
          return Spell()

class EnchantedRoom(Room):
      def __init__(self, roomNo,aSpell):
          super(EnchantedRoom,self).__init__(roomNo)
          print("the spell is: ",aSpell)
          
class Spell():
      def __repr__(self):
          return '"A hard-coded Spell"'

class DoorNeedingSpell(Door):
      def __init__(self,r1,r2):
          super(DoorNeedingSpell,self).__init__(r1,r2)
          self.spell = Spell()
      def Enter(self):
          print("    +This door needs a spell....",self.spell)
          if self._isOpen: print("    ***You have passed through the door...")
          else: print("  ****This door needs tO be opened before you can pass through it....")
    
"""class MazeGame():
    def CreatMaze(self,factory= MazeFactory):
        aMaze = factory.MakeMaze()
        r1 = factory.MakeRoom(1)
        r2 = factory.MakeRoom(2)
        aDoor = factory.MakeDoor(r1,r2)
        
        aMaze.AddRoom(r1)
        aMaze.AddRoom(r2)
        
        r1.SetSide(Direction.North.value,factory.MakeWall())
        r1.SetSide(Direction.East.value,aDoor)
        r1.SetSide(Direction.South.value,factory.MakeWall())
        r1.SetSide(Direction.West.value,factory.MakeWall())
        
        r2.SetSide(Direction(0).value,factory.MakeWall())
        r2.SetSide(Direction(1).value,factory.MakeWall())
        r2.SetSide(Direction(2).value,factory.MakeWall())
        r2.SetSide(Direction(3).value,aDoor)
        
        return aMaze"""
    
class MazeGame():
    def CreatMaze(self,builder):
        builder.BuildMaze()
        builder.BuildRoom(1)
        builder.BuildRoom(2)
        builder.BuildDoor(1,2)
        
        return builder.GetMaze()
    
    def CreatComplexMaze(self,builder):
        builder.BuildRoom(1)
        #....
        builder.BuildRoom(1001)
        
        return builder.GetMaze()
class StandardMazeBuilder(MazeBuilder) :
    def __init__(self):
        self._currentMaze = None
        
    def BuildMaze(self):
        self._currentMaze = Maze()
        
    def BuildRoom(self, n):
        try: self._currentMaze.RoomNo(n)
        except:
            print("Room {} does not exist - building this room".format(n))
            room = Room(n)
            self._currentMaze.AddRoom(room)
            
            room.SetSide(Direction.North.value, Wall())
            room.SetSide(Direction.East.value, Wall())
            room.SetSide(Direction.West.value, Wall())
            room.SetSide(Direction.South.value, Wall())
            
    def BuildDoor(self, n1, n2):
      r1 =  self._currentMaze.RoomNo(n1)
      r2 = self._currentMaze.RoomNo(n2)
      d = Door(r1,r2)
      
      r1.SetSide(self.CommonWall(r1,r2),d)
      r2.SetSide(self.CommonWall(r1,r2),d)
      
      print()
      for side in range(4):
          if "Door" in str(r1._sides[side]):
              print("Room1:",r1._sides[side],Direction(side))
          if "Door" in  str(r2._sides[side]):
              print("Room2:",r2._sides[side],Direction(side))
              
    def GetMaze(self):
        return self._currentMaze
    
    def CommonWall(self,aRoom,anotherRoom):
        if aRoom._roomNumber < anotherRoom._roomNumber:
            return Direction.East.value
        else:
            return Direction.West.value
        
if __name__ == "__main__":
    
    
    """def find_maze_rooms(maze_obj):
        maze_rooms = []
        for room_number in range(5):
            try:
                room  = maze_obj.RoomNo(room_number)
                print("\n^^^ Maze Has Room: {}".format(room_number,room))
                print("     Entering the room....")
                room.Enter()
                maze_rooms.append(room)
                for idx in range(4):
                    side = room.GetSide(idx)
                    side_str = str(side.__class__).replace("<<class '__main__.","").replace("'>","")
                    print("  Room: {}, {:<15s}, Type: {}".format(room_number,Direction(idx),side_str))
                    print("  Trying to Enter: ",Direction(idx))
                    side.Enter()
                    if "Door" in side_str:
                        door = side
                        if not door._isOpen:
                            print("   ***Opening the Door...")
                            door._isOpen = True
                            door.Enter()
                        print("\t",door)
                        other_room = door.OtherSideFrom(room)
                        print("\nOn the Other side of the door is room: {}\n".format(other_room))
                        
            except KeyError:
                print("No Room: ",room_number)
        num_of_rooms = len(maze_rooms)
        print("\nThere are {} rooms in the maze.".format(num_of_rooms))
        
        print("Both doors are the same object and they are on the west and east side of the two room") """  
    
print("*" * 21)
print("*** The Maze Game ***")
print("*" * 21)    

"""factory = MazeFactory()
print(factory)

maze_obj = MazeGame().CreatMaze()
find_maze_rooms(maze_obj)

print("*" * 21)
print("*** The Enchated Maze Game ***")
print("*" * 21)  

factory = EnchantedMazeFactory()
print(factory)

maze_obj = MazeGame().CreatMaze()
find_maze_rooms(maze_obj)"""

maze = Maze
game = MazeGame()
builder = StandardMazeBuilder()

game.CreatMaze(builder)
maze = builder.GetMaze()




            
        
          
    
        
        
        
      