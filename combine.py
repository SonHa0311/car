import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import math
import sys
import os
import heapq
import socket
from heapdict import heapdict
from scipy.spatial import KDTree
import reeds_shepp as rsCurve
from pyapriltags import Detector
from scipy.interpolate import LinearNDInterpolator, splprep, splev
from scipy.spatial.distance import cdist

# Cấu hình kết nối TCP đến File 3
file3_ip = "127.0.0.1"  # Sử dụng localhost để kết nối đến file3
file3_port = 5002

# Kết nối TCP đến File 3
def connect_to_file3():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # Đặt timeout 5 giây
        sock.connect((file3_ip, file3_port))
        print(f"Đã kết nối thành công đến File 3 tại {file3_ip}:{file3_port}")
        return sock
    except socket.timeout:
        print(f"Không thể kết nối đến File 3: Timeout")
        return None
    except ConnectionRefusedError:
        print(f"Không thể kết nối đến File 3: Connection refused")
        return None
    except Exception as e:
        print(f"Lỗi kết nối đến File 3: {e}")
        return None

# Gửi lệnh điều khiển qua TCP
def send_tcp_command(sock, command):
    if sock is None:
        print(f"Không thể gửi lệnh '{command}': Socket không kết nối")
        return False
    
    try:
        sock.sendall((command + "\n").encode())
        print(f"Đã gửi lệnh: {command}")
        return True
    except Exception as e:
        print(f"Lỗi gửi lệnh '{command}': {e}")
        return False

# ===== Các lớp và hàm từ file2.py =====

class Car:
    maxSteerAngle = 0.45
    steerPresion = 10
    wheelBase = 5
    axleToFront = 6
    axleToBack = 4
    width = 5

class Cost:
    reverse = 10
    directionChange = 150
    steerAngle = 1
    steerAngleChange = 5
    hybridCost = 50

class Node:
    def __init__(self, gridIndex, traj, steeringAngle, direction, cost, parentIndex):
        self.gridIndex = gridIndex         # grid block x, y, yaw index
        self.traj = traj                   # trajectory x, y  of a simulated node
        self.steeringAngle = steeringAngle # steering angle throughout the trajectory
        self.direction = direction         # direction throughout the trajectory
        self.cost = cost                   # node cost
        self.parentIndex = parentIndex     # parent node index

class HolonomicNode:
    def __init__(self, gridIndex, cost, parentIndex):
        self.gridIndex = gridIndex
        self.cost = cost
        self.parentIndex = parentIndex

class MapParameters:
    def __init__(self, mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX, obstacleY):
        self.mapMinX = mapMinX               # map min x coordinate(0)
        self.mapMinY = mapMinY               # map min y coordinate(0)
        self.mapMaxX = mapMaxX               # map max x coordinate
        self.mapMaxY = mapMaxY               # map max y coordinate
        self.xyResolution = xyResolution     # grid block length
        self.yawResolution = yawResolution   # grid block possible yaws
        self.ObstacleKDTree = ObstacleKDTree # KDTree representating obstacles
        self.obstacleX = obstacleX           # Obstacle x coordinate list
        self.obstacleY = obstacleY           # Obstacle y coordinate list

def calculateMapParameters(obstacleX, obstacleY, xyResolution, yawResolution):
        
        # calculate min max map grid index based on obstacles in map
        mapMinX = round(min(obstacleX) / xyResolution)
        mapMinY = round(min(obstacleY) / xyResolution)
        mapMaxX = round(max(obstacleX) / xyResolution)
        mapMaxY = round(max(obstacleY) / xyResolution)

        # create a KDTree to represent obstacles
        ObstacleKDTree = KDTree([[x, y] for x, y in zip(obstacleX, obstacleY)])

        return MapParameters(mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX, obstacleY)  

def index(Node):
    # Index is a tuple consisting grid index, used for checking if two nodes are near/same
    return tuple([Node.gridIndex[0], Node.gridIndex[1], Node.gridIndex[2]])

def motionCommands():

    # Motion commands for a Non-Holonomic Robot like a Car or Bicycle (Trajectories using Steer Angle and Direction)
    direction = 1
    motionCommand = []
    for i in np.arange(Car.maxSteerAngle, -(Car.maxSteerAngle + Car.maxSteerAngle/Car.steerPresion), -Car.maxSteerAngle/Car.steerPresion):
        motionCommand.append([i, direction])
        motionCommand.append([i, -direction])
    return motionCommand

def holonomicMotionCommands():

    # Action set for a Point/Omni-Directional/Holonomic Robot (8-Directions)
    holonomicMotionCommand = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
    return holonomicMotionCommand

def kinematicSimulationNode(currentNode, motionCommand, mapParameters, simulationLength=4, step = 0.8 ):
    # Simulate node using given current Node and Motion Commands
    traj = []
    angle = rsCurve.pi_2_pi(currentNode.traj[-1][2] + motionCommand[1] * step / Car.wheelBase * math.tan(motionCommand[0]))
    traj.append([currentNode.traj[-1][0] + motionCommand[1] * step * math.cos(angle),
                currentNode.traj[-1][1] + motionCommand[1] * step * math.sin(angle),
                rsCurve.pi_2_pi(angle + motionCommand[1] * step / Car.wheelBase * math.tan(motionCommand[0]))])
    for i in range(int((simulationLength/step))-1):
        traj.append([traj[i][0] + motionCommand[1] * step * math.cos(traj[i][2]),
                    traj[i][1] + motionCommand[1] * step * math.sin(traj[i][2]),
                    rsCurve.pi_2_pi(traj[i][2] + motionCommand[1] * step / Car.wheelBase * math.tan(motionCommand[0]))])

    # Find grid index
    gridIndex = [round(traj[-1][0]/mapParameters.xyResolution), \
                 round(traj[-1][1]/mapParameters.xyResolution), \
                 round(traj[-1][2]/mapParameters.yawResolution)]

    # Check if node is valid
    if not isValid(traj, gridIndex, mapParameters):
        return None

    # Calculate Cost of the node
    cost = simulatedPathCost(currentNode, motionCommand, simulationLength)

    return Node(gridIndex, traj, motionCommand[0], motionCommand[1], cost, index(currentNode))

def reedsSheppNode(currentNode, goalNode, mapParameters):

    # Get x, y, yaw of currentNode and goalNode
    startX, startY, startYaw = currentNode.traj[-1][0], currentNode.traj[-1][1], currentNode.traj[-1][2]
    goalX, goalY, goalYaw = goalNode.traj[-1][0], goalNode.traj[-1][1], goalNode.traj[-1][2]

    # Instantaneous Radius of Curvature
    radius = math.tan(Car.maxSteerAngle)/Car.wheelBase

    #  Find all possible reeds-shepp paths between current and goal node
    reedsSheppPaths = rsCurve.calc_all_paths(startX, startY, startYaw, goalX, goalY, goalYaw, radius, 1)

    # Check if reedsSheppPaths is empty
    if not reedsSheppPaths:
        return None

    # Find path with lowest cost considering non-holonomic constraints
    costQueue = heapdict()
    for path in reedsSheppPaths:
        costQueue[path] = reedsSheppCost(currentNode, path)

    # Find first path in priority queue that is collision free
    while len(costQueue)!=0:
        path = costQueue.popitem()[0]
        traj=[]
        traj = [[path.x[k],path.y[k],path.yaw[k]] for k in range(len(path.x))]
        if not collision(traj, mapParameters):
            cost = reedsSheppCost(currentNode, path)
            return Node(goalNode.gridIndex ,traj, None, None, cost, index(currentNode))
            
    return None

def isValid(traj, gridIndex, mapParameters):

    # Check if Node is out of map bounds
    if gridIndex[0]<=mapParameters.mapMinX or gridIndex[0]>=mapParameters.mapMaxX or \
       gridIndex[1]<=mapParameters.mapMinY or gridIndex[1]>=mapParameters.mapMaxY:
        return False

    # Check if Node is colliding with an obstacle
    if collision(traj, mapParameters):
        return False
    return True

def collision(traj, mapParameters):

    carRadius = (Car.axleToFront + Car.axleToBack)/2 + 1
    dl = (Car.axleToFront - Car.axleToBack)/2
    for i in traj:
        cx = i[0] + dl * math.cos(i[2])
        cy = i[1] + dl * math.sin(i[2])
        pointsInObstacle = mapParameters.ObstacleKDTree.query_ball_point([cx, cy], carRadius)

        if not pointsInObstacle:
            continue

        for p in pointsInObstacle:
            xo = mapParameters.obstacleX[p] - cx
            yo = mapParameters.obstacleY[p] - cy
            dx = xo * math.cos(i[2]) + yo * math.sin(i[2])
            dy = -xo * math.sin(i[2]) + yo * math.cos(i[2])

            if abs(dx) < carRadius and abs(dy) < Car.width / 2 + 1:
                return True

    return False

def reedsSheppCost(currentNode, path):

    # Previos Node Cost
    cost = currentNode.cost

    # Distance cost
    for i in path.lengths:
        if i >= 0:
            cost += 1
        else:
            cost += abs(i) * Cost.reverse

    # Direction change cost
    for i in range(len(path.lengths)-1):
        if path.lengths[i] * path.lengths[i+1] < 0:
            cost += Cost.directionChange

    # Steering Angle Cost
    for i in path.ctypes:
        # Check types which are not straight line
        if i!="S":
            cost += Car.maxSteerAngle * Cost.steerAngle

    # Steering Angle change cost
    turnAngle=[0.0 for _ in range(len(path.ctypes))]
    for i in range(len(path.ctypes)):
        if path.ctypes[i] == "R":
            turnAngle[i] = - Car.maxSteerAngle
        if path.ctypes[i] == "WB":
            turnAngle[i] = Car.maxSteerAngle

    for i in range(len(path.lengths)-1):
        cost += abs(turnAngle[i+1] - turnAngle[i]) * Cost.steerAngleChange

    return cost

def simulatedPathCost(currentNode, motionCommand, simulationLength):

    # Previos Node Cost
    cost = currentNode.cost

    # Distance cost
    if motionCommand[1] == 1:
        cost += simulationLength 
    else:
        cost += simulationLength * Cost.reverse

    # Direction change cost
    if currentNode.direction != motionCommand[1]:
        cost += Cost.directionChange

    # Steering Angle Cost
    cost += motionCommand[0] * Cost.steerAngle

    # Steering Angle change cost
    cost += abs(motionCommand[0] - currentNode.steeringAngle) * Cost.steerAngleChange

    return cost

def eucledianCost(holonomicMotionCommand):
    # Compute Eucledian Distance between two nodes
    return math.hypot(holonomicMotionCommand[0], holonomicMotionCommand[1])

def holonomicNodeIndex(HolonomicNode):
    # Index is a tuple consisting grid index, used for checking if two nodes are near/same
    return tuple([HolonomicNode.gridIndex[0], HolonomicNode.gridIndex[1]])

def obstaclesMap(obstacleX, obstacleY, xyResolution):

    # Compute Grid Index for obstacles
    obstacleX = [round(x / xyResolution) for x in obstacleX]
    obstacleY = [round(y / xyResolution) for y in obstacleY]

    # Set all Grid locations to No Obstacle
    obstacles =[[False for i in range(max(obstacleY))]for i in range(max(obstacleX))]

    # Set Grid Locations with obstacles to True
    for x in range(max(obstacleX)):
        for y in range(max(obstacleY)):
            for i, j in zip(obstacleX, obstacleY):
                if math.hypot(i-x, j-y) <= 1/2:
                    obstacles[i][j] = True
                    break

    return obstacles

def holonomicNodeIsValid(neighbourNode, obstacles, mapParameters):

    # Check if Node is out of map bounds
    if neighbourNode.gridIndex[0]<= mapParameters.mapMinX or \
       neighbourNode.gridIndex[0]>= mapParameters.mapMaxX or \
       neighbourNode.gridIndex[1]<= mapParameters.mapMinY or \
       neighbourNode.gridIndex[1]>= mapParameters.mapMaxY:
        return False

    # Check if Node on obstacle
    if obstacles[neighbourNode.gridIndex[0]][neighbourNode.gridIndex[1]]:
        return False

    return True

def holonomicCostsWithObstacles(goalNode, mapParameters):

    gridIndex = [round(goalNode.traj[-1][0]/mapParameters.xyResolution), round(goalNode.traj[-1][1]/mapParameters.xyResolution)]
    gNode =HolonomicNode(gridIndex, 0, tuple(gridIndex))

    obstacles = obstaclesMap(mapParameters.obstacleX, mapParameters.obstacleY, mapParameters.xyResolution)

    holonomicMotionCommand = holonomicMotionCommands()

    openSet = {holonomicNodeIndex(gNode): gNode}
    closedSet = {}

    priorityQueue =[]
    heapq.heappush(priorityQueue, (gNode.cost, holonomicNodeIndex(gNode)))

    while True:
        if not openSet:
            break

        _, currentNodeIndex = heapq.heappop(priorityQueue)
        currentNode = openSet[currentNodeIndex]
        openSet.pop(currentNodeIndex)
        closedSet[currentNodeIndex] = currentNode

        for i in range(len(holonomicMotionCommand)):
            neighbourNode = HolonomicNode([currentNode.gridIndex[0] + holonomicMotionCommand[i][0],\
                                      currentNode.gridIndex[1] + holonomicMotionCommand[i][1]],\
                                      currentNode.cost + eucledianCost(holonomicMotionCommand[i]), currentNodeIndex)

            if not holonomicNodeIsValid(neighbourNode, obstacles, mapParameters):
                continue

            neighbourNodeIndex = holonomicNodeIndex(neighbourNode)

            if neighbourNodeIndex not in closedSet:            
                if neighbourNodeIndex in openSet:
                    if neighbourNode.cost < openSet[neighbourNodeIndex].cost:
                        openSet[neighbourNodeIndex].cost = neighbourNode.cost
                        openSet[neighbourNodeIndex].parentIndex = neighbourNode.parentIndex
                        # heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))
                else:
                    openSet[neighbourNodeIndex] = neighbourNode
                    heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))

    holonomicCost = [[np.inf for i in range(max(mapParameters.obstacleY))]for i in range(max(mapParameters.obstacleX))]

    for nodes in closedSet.values():
        holonomicCost[nodes.gridIndex[0]][nodes.gridIndex[1]]=nodes.cost

    return holonomicCost

def create_map():
    obstacleX, obstacleY = [], []

    # Scale down the original dimensions (5000, 3000) to fit the grid
    scale = 10  # This means 1 grid unit = 100 pixels in original map
    map_width = 500  # 5000/100
    map_height = 300  # 3000/100

    # Outer straight lines
    # bottom horizontal line
    for i in range(132, map_width - 132):
        obstacleX.append(i)
        obstacleY.append(0)

    # top horizontal line
    for i in range(132, map_width - 132):
        obstacleX.append(i)
        obstacleY.append(map_height)

    # Left vertical line
    for i in range(132, map_height - 132):
        obstacleX.append(0)
        obstacleY.append(i)

    # Right vertical line
    for i in range(132, map_height - 132):
        obstacleX.append(map_width)
        obstacleY.append(i)

    # Inner straight lines
    # Top inner horizontal line
    for i in range(60, 145):
        obstacleX.append(i)
        obstacleY.append(120)
    for i in range(205, 220):
        obstacleX.append(i)
        obstacleY.append(120)

    for i in range(280, 295):
        obstacleX.append(i)
        obstacleY.append(120)
    for i in range(325, 440):
        obstacleX.append(i)
        obstacleY.append(120)

    # chuong ngang 600x312
    for i in range(145, 205):
        obstacleX.append(i)
        obstacleY.append(88)
    for i in range(88, 120):
        obstacleX.append(145)
        obstacleY.append(i)
    for i in range(88, 120):
        obstacleX.append(205)
        obstacleY.append(i)

    # chuong doc 300x550
    for i in range(295, 325):
        obstacleX.append(i)
        obstacleY.append(65)
    for i in range(65, 120):
        obstacleX.append(295)
        obstacleY.append(i)
    for i in range(65, 120):
        obstacleX.append(325)
        obstacleY.append(i)

    # Bottom inner horizontal line
    for i in range(60, 220):
        obstacleX.append(i)
        obstacleY.append(180)
    for i in range(280, 440):
        obstacleX.append(i)
        obstacleY.append(180)

    # Left inner vertical line
    for i in range(60, 120):
        obstacleX.append(220)
        obstacleY.append(i)
    for i in range(180, 240):
        obstacleX.append(220)
        obstacleY.append(i)

    # Right inner vertical line
    for i in range(60, 120):
        obstacleX.append(280)
        obstacleY.append(i)
    for i in range(180, 240):
        obstacleX.append(280)
        obstacleY.append(i)

    # Function to add corner points
    def add_corner_points(radius, center_x, center_y, start_angle, end_angle):
        for angle in range(start_angle, end_angle + 1):
            x = radius * math.cos(math.radians(angle)) + center_x
            y = radius * math.sin(math.radians(angle)) + center_y
            obstacleX.append(round(x))
            obstacleY.append(round(y))

    # Outer corners
    outer_radius = 132
    add_corner_points(outer_radius, outer_radius, outer_radius, 180, 270)
    add_corner_points(outer_radius, map_width - outer_radius, outer_radius, 270, 360)
    add_corner_points(outer_radius, outer_radius, map_height - outer_radius, 90, 180)
    add_corner_points(outer_radius, map_width - outer_radius, map_height - outer_radius, 0, 90)

    return obstacleX, obstacleY

def backtrack(startNode, goalNode, closedSet, plt):

    # Goal Node data
    startNodeIndex= index(startNode)
    currentNodeIndex = goalNode.parentIndex
    currentNode = closedSet[currentNodeIndex]
    x=[]
    y=[]
    yaw=[]

    # Iterate till we reach start node from goal node
    while currentNodeIndex != startNodeIndex:
        a, b, c = zip(*currentNode.traj)
        x += a[::-1] 
        y += b[::-1] 
        yaw += c[::-1]
        currentNodeIndex = currentNode.parentIndex
        currentNode = closedSet[currentNodeIndex]
    return x[::-1], y[::-1], yaw[::-1]

def run(s, g, mapParameters, plt):

    # Compute Grid Index for start and Goal node
    sGridIndex = [round(s[0] / mapParameters.xyResolution), \
                  round(s[1] / mapParameters.xyResolution), \
                  round(s[2]/mapParameters.yawResolution)]
    gGridIndex = [round(g[0] / mapParameters.xyResolution), \
                  round(g[1] / mapParameters.xyResolution), \
                  round(g[2]/mapParameters.yawResolution)]

    # Generate all Possible motion commands to car
    motionCommand = motionCommands()

    # Create start and end Node
    startNode = Node(sGridIndex, [s], 0, 1, 0 , tuple(sGridIndex))
    goalNode = Node(gGridIndex, [g], 0, 1, 0, tuple(gGridIndex))

    # Find Holonomic Heuristric
    holonomicHeuristics = holonomicCostsWithObstacles(goalNode, mapParameters)

    # Add start node to open Set
    openSet = {index(startNode):startNode}
    closedSet = {}

    # Create a priority queue for acquiring nodes based on their cost's
    costQueue = heapdict()

    # Add start mode into priority queue
    costQueue[index(startNode)] = max(startNode.cost , Cost.hybridCost * holonomicHeuristics[startNode.gridIndex[0]][startNode.gridIndex[1]])
    counter = 0

    # Run loop while path is found or open set is empty
    while True:
        counter +=1
        # Check if openSet is empty, if empty no solution available
        if not openSet:
            return None

        # Get first node in the priority queue
        currentNodeIndex = costQueue.popitem()[0]
        currentNode = openSet[currentNodeIndex]

        # Revove currentNode from openSet and add it to closedSet
        openSet.pop(currentNodeIndex)
        closedSet[currentNodeIndex] = currentNode


        # Get Reed-Shepp Node if available
        rSNode = reedsSheppNode(currentNode, goalNode, mapParameters)

        # Id Reeds-Shepp Path is found exit
        if rSNode:
            closedSet[index(rSNode)] = rSNode
            break

        # USED ONLY WHEN WE DONT USE REEDS-SHEPP EXPANSION OR WHEN START = GOAL
        if currentNodeIndex == index(goalNode):
            print("Path Found")
            print(currentNode.traj[-1])
            break

        # Get all simulated Nodes from current node
        for i in range(len(motionCommand)):
            simulatedNode = kinematicSimulationNode(currentNode, motionCommand[i], mapParameters)

            # Check if path is within map bounds and is collision free
            if not simulatedNode:
                continue

            # Draw Simulated Node
            x,y,z =zip(*simulatedNode.traj)
            plt.plot(x, y, linewidth=0.3, color='g')

            # Check if simulated node is already in closed set
            simulatedNodeIndex = index(simulatedNode)
            if simulatedNodeIndex not in closedSet: 

                # Check if simulated node is already in open set, if not add it open set as well as in priority queue
                if simulatedNodeIndex not in openSet:
                    openSet[simulatedNodeIndex] = simulatedNode
                    costQueue[simulatedNodeIndex] = max(simulatedNode.cost , Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]])
                else:
                    if simulatedNode.cost < openSet[simulatedNodeIndex].cost:
                        openSet[simulatedNodeIndex] = simulatedNode
                        costQueue[simulatedNodeIndex] = max(simulatedNode.cost , Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]])
    
    # Backtrack
    x, y, yaw = backtrack(startNode, goalNode, closedSet, plt)

    return x, y, yaw

def drawCar(x, y, yaw, color='black'):
    car = np.array([[-Car.axleToBack, -Car.axleToBack, Car.axleToFront, Car.axleToFront, -Car.axleToBack],
                    [Car.width / 2, -Car.width / 2, -Car.width / 2, Car.width / 2, Car.width / 2]])

    rotationZ = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])
    car = np.dot(rotationZ, car)
    car += np.array([[x], [y]])
    plt.plot(car[0, :], car[1, :], color)

# ===== Các hàm từ file1.py =====

# Tọa độ pixel và tọa độ thực (map)
pts_pixel = np.array([[10, 10], [1900, 10], [10, 1000], [1900, 1000]])
pts_real = np.array([[0, 0], [4.94, 0], [0, 3.2], [4.94, 3.2]])

interpolator_x = LinearNDInterpolator(pts_pixel, pts_real[:, 0])
interpolator_y = LinearNDInterpolator(pts_pixel, pts_real[:, 1])

def pixel_to_real(pixel_coords):
    pixel_coords = np.array(pixel_coords, dtype='float32')
    real_x = interpolator_x(pixel_coords)
    real_y = interpolator_y(pixel_coords)
    # Convert numpy arrays to scalar values
    return float(real_x), float(real_y)

def calculate_angle(p1, p2):
    """ Tính góc giữa hai điểm so với trục x dương """
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi  # Đổi sang độ

# 📌 Hàm tải dữ liệu các đường biên đã lưu
def load_lines():
    try:
        with open("lines_data2.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Không tìm thấy file dữ liệu!")
        return []

# 📌 Nội suy spline cho đường biên
def interpolate_spline(line, num_points=5000):
    line = np.array(line)
    tck, u = splprep([line[:, 0], line[:, 1]], s=0)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    return np.vstack((x_new, y_new)).T

# 📌 Tìm điểm giao của tia từ trung tâm với đường biên
def find_intersection(ray_origin, ray_angle, line_points):
    direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
    ray_end = ray_origin + direction * 1000
    distances = cdist(line_points, [ray_end])
    closest_index = np.argmin(distances)
    return line_points[closest_index]

# 📌 Tính đường trung tâm từ 2 đường biên
def calculate_center_line(line1, line2, center, num_rays=360):
    line1_interp = interpolate_spline(line1)
    line2_interp = interpolate_spline(line2)
    center_points = []
    for angle in np.linspace(0, 2 * np.pi, num_rays):
        p1 = find_intersection(center, angle, line1_interp)
        p2 = find_intersection(center, angle, line2_interp)
        center_point = (p1 + p2) / 2
        center_points.append(center_point)
    return np.array(center_points)

# 📌 Vẽ đường trung tâm và hiển thị AprilTag
def draw_center_line(tag_positions):
    image = cv2.imread("captured_birdseye.png", cv2.IMREAD_GRAYSCALE)
    # cropped_image = image[33:650, 198:1270]
    cropped_image = image[0:1080, 0:1920]

    _, binary_map = cv2.threshold(cropped_image, 175, 255, cv2.THRESH_BINARY_INV)

    all_lines = load_lines()
    if len(all_lines) < 2:
        print("Cần ít nhất 2 đường để vẽ trung tâm!")
        return

    line1, line2 = np.array(all_lines[0]), np.array(all_lines[1])
    center = np.mean(np.vstack((line1, line2)), axis=0)
    center_line = calculate_center_line(line1, line2, center)

    fig, ax = plt.subplots()
    # ax.imshow(binary_map, cmap='gray', extent=[198, 1270, 650, 33])
    ax.imshow(binary_map, cmap='gray', extent=[0, 1980, 1080, 0])

    # Vẽ đường biên
    line1_smooth = interpolate_spline(line1)
    line2_smooth = interpolate_spline(line2)
    ax.plot(line1_smooth[:, 0], line1_smooth[:, 1], color='blue', label="Đường 1")
    ax.plot(line2_smooth[:, 0], line2_smooth[:, 1], color='red', label="Đường 2")
    ax.plot(center_line[:, 0], center_line[:, 1], color='green', linestyle='--', linewidth=2, label="Đường trung tâm")

    # Vẽ các AprilTag
    colors = {1: 'cyan', 2: 'magenta', 3: 'yellow', 4: 'lime', 5: 'orange', 13: 'purple', 12345: 'pink'}
    for tag_id, positions in tag_positions.items():
        if len(positions) > 0:
            positions = np.array(positions)
            ax.scatter(positions[:, 0], positions[:, 1], color=colors[tag_id], label=f"Tag {tag_id}")
            ax.plot(positions[:, 0], positions[:, 1], color=colors[tag_id], linestyle='-', linewidth=2)

    ax.legend()
    plt.show()

# ===== Hàm kết hợp chức năng của cả hai file =====

def process_video_and_plan_path(video_source=0):
    """Xử lý video stream, phát hiện AprilTag và lập kế hoạch đường đi"""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Không thể mở video source {video_source}")
        return

    detector = Detector(families="tag25h9")
    tag_positions = {}
    
    # Biến lưu vị trí hiện tại của xe
    current_car_x = 0
    current_car_y = 0
    current_car_yaw = 0
    
    # Biến để kiểm tra xem đã tìm thấy đường đi chưa
    path_found = False
    path_x = None
    path_y = None
    path_yaw = None
    
    # Biến để kiểm tra xem đã gửi lệnh điều khiển chưa
    control_commands_sent = False
    
    # Biến để lưu trữ các lệnh điều khiển
    control_commands = []
    
    print("Đang xử lý video stream...")
    print("Nhấn 'q' để thoát")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ video stream")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]

        pts_src = np.float32([[0, height], [width, height], [width * 0.9, height * 0.1], [width * 0.1, height * 0.1]])
        pts_dst = np.float32([[0, height], [width, height], [width, 0], [0, 0]])
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        birdseye_frame = cv2.warpPerspective(frame, matrix, (width, height))
        birdseye_gray = cv2.warpPerspective(gray, matrix, (width, height))

        tags = detector.detect(birdseye_gray)
        car_x, car_y, car_yaw = None, None, None

        for tag in tags:
            tag_x, tag_y = map(int, tag.center)
            real_x, real_y = pixel_to_real((tag_x, tag_y))

            if tag.tag_id == 13:
                top_left = tag.corners[0]  
                top_right = tag.corners[1]  
                mid_x = (top_left[0] + top_right[0]) / 2
                mid_y = (top_left[1] + top_right[1]) / 2
                car_yaw = calculate_angle((tag_x, tag_y), (mid_x, mid_y))
                car_x, car_y = real_x, real_y

                arrow_x = int(tag_x + 30 * np.cos(car_yaw * np.pi / 180))
                arrow_y = int(tag_y + 30 * np.sin(car_yaw * np.pi / 180))
                cv2.arrowedLine(birdseye_frame, (tag_x, tag_y), (arrow_x, arrow_y), (0, 255, 0), 2, tipLength=0.3)

            cv2.circle(birdseye_frame, (tag_x, tag_y), 5, (0, 0, 255), -1)
            cv2.putText(birdseye_frame, f"ID: {tag.tag_id} ({float(real_x):.2f}, {float(real_y):.2f})",
                        (tag_x + 10, tag_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if car_x is not None and car_y is not None and car_yaw is not None:
            # Cập nhật vị trí hiện tại của xe
            current_car_x = car_x
            current_car_y = car_y
            current_car_yaw = car_yaw
            
            # Nếu chưa tìm thấy đường đi, thực hiện lập kế hoạch đường đi
            if not path_found:
                print(f"Phát hiện xe tại: x={car_x:.2f}, y={car_y:.2f}, yaw={car_yaw:.2f}")
                
                # Lập kế hoạch đường đi
                path_x, path_y, path_yaw = plan_path(car_x, car_y, car_yaw)
                
                if path_x is not None:
                    path_found = True
                    print("Đã tìm thấy đường đi!")
                    
                    # Tạo các lệnh điều khiển
                    control_commands = generate_control_commands(path_x, path_y, path_yaw)
                    print(f"Đã tạo {len(control_commands)} lệnh điều khiển")
            
            # Nếu đã tìm thấy đường đi và chưa gửi lệnh điều khiển
            if path_found and not control_commands_sent:
                # Gửi lệnh điều khiển
                send_control_commands(control_commands)
                control_commands_sent = True
                print("Đã gửi lệnh điều khiển")

        cv2.imshow("AprilTag Detection", frame)
        cv2.imshow("Bird's Eye View", birdseye_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Vẽ đường đi và vị trí của xe
    if path_found:
        draw_path_and_car(path_x, path_y, path_yaw, current_car_x, current_car_y, current_car_yaw)

def plan_path(x, y, theta):
    """Lập kế hoạch đường đi từ vị trí hiện tại đến mục tiêu"""
    # Chuyển đổi theta từ độ sang radian
    theta_rad = np.radians(theta)
    
    # Vị trí bắt đầu
    s = [x * 100, y * 100, theta_rad]
    
    # Vị trí mục tiêu
    g = [450, 150, np.deg2rad(180)]
    
    # Lấy dữ liệu bản đồ chướng ngại vật
    obstacleX, obstacleY = create_map()
    
    # Tính toán tham số bản đồ
    mapParameters = calculateMapParameters(obstacleX, obstacleY, 4, np.deg2rad(15.0))
    
    # Chạy thuật toán Hybrid A*
    x, y, yaw = run(s, g, mapParameters, None)
    
    return x, y, yaw

def generate_control_commands(x, y, yaw):
    """Tạo các lệnh điều khiển dựa trên đường đi"""
    commands = []
    
    for k in range(len(x)):
        # Tính toán hướng di chuyển
        if k < len(x) - 1:
            direct_ang = np.arctan2(y[k+1] - y[k], x[k+1] - x[k])
            direct_diff = np.degrees(rsCurve.pi_2_pi(direct_ang - yaw[k]))
            yaw_diff = np.degrees(rsCurve.pi_2_pi(yaw[k+1] - yaw[k]))
            
            # Điều khiển tốc độ & góc lái
            if -90 <= direct_diff <= 90:
                commands.append("M-6000")  # Tiến
                if yaw_diff < -1:
                    commands.append("S110")  # Rẽ trái gấp
                elif yaw_diff > 1:
                    commands.append("S50")   # Rẽ phải gấp
                else:
                    commands.append("S80")   # Thẳng
            else:
                commands.append("M+6000")  # Lùi
                if yaw_diff < -1:
                    commands.append("S50")   # Rẽ phải gấp
                elif yaw_diff > 1:
                    commands.append("S110")  # Rẽ trái gấp
                else:
                    commands.append("S80")   # Thẳng
            
            # Kiểm tra hướng tiếp theo
            if k < len(yaw) - 2:
                direct_ang_next = np.arctan2(y[k+2] - y[k+1], x[k+2] - x[k+1])
                direct_diff_next = np.degrees(direct_ang_next - direct_ang)
                if direct_diff_next >= 90:
                    commands.append("M-0")  # Dừng
                    commands.append("M-0")  # Dừng
    
    # Thêm lệnh dừng cuối cùng
    commands.append("M-0")
    commands.append("S80")
    
    return commands

def send_control_commands(commands):
    """Gửi các lệnh điều khiển qua TCP đến file3.py"""
    # Kết nối đến File 3
    sock = connect_to_file3()
    if sock is None:
        print("Không thể kết nối đến File 3, không thể gửi lệnh điều khiển")
        return

    try:
        for command in commands:
            send_tcp_command(sock, command)
            time.sleep(0)  # Không cần đợi giữa các lệnh
    finally:
        sock.close()
        print("Đã đóng kết nối TCP")

def draw_path_and_car(path_x, path_y, path_yaw, car_x, car_y, car_yaw):
    """Vẽ đường đi và vị trí của xe"""
    plt.figure(figsize=(10, 8))
    
    # Vẽ đường đi
    plt.plot(path_x, path_y, 'b-', linewidth=2, label='Đường đi')
    
    # Vẽ vị trí hiện tại của xe
    drawCar(car_x * 100, car_y * 100, np.radians(car_yaw), 'r')
    
    # Vẽ vị trí mục tiêu
    plt.plot(450, 150, 'g*', markersize=15, label='Mục tiêu')
    
    # Vẽ bản đồ chướng ngại vật
    obstacleX, obstacleY = create_map()
    plt.plot(obstacleX, obstacleY, 'k.', markersize=1, label='Chướng ngại vật')
    
    plt.grid(True)
    plt.legend()
    plt.title('Đường đi và vị trí xe')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

# Hàm chính
def main():
    # Sử dụng camera mặc định (index 0)
    # Có thể thay đổi index để sử dụng camera khác
    # Hoặc sử dụng URL của IP camera, ví dụ: "rtsp://username:password@ip_address:port/stream"
    video_source = 0  # Thay đổi giá trị này để sử dụng camera khác hoặc IP camera
    
    # Xử lý video stream và lập kế hoạch đường đi
    process_video_and_plan_path(video_source)

if __name__ == "__main__":
    main() 
