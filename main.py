#!/usr/bin/python
import yaml
import math
import numpy as np
from matplotlib import pyplot as plt

configFile = "./settings.yaml"

def main():
    conf = None
    with open(configFile, "r") as stream:
        try:
            conf = yaml.safe_load(stream)
        except:
            exit(1)
    
    mat = calcMatrix(testFunc, conf['graph'])

    vec = generateVectors(mat, conf['graph'])
    rvec = rotateVectors(vec, conf['graph'])

    gcode = generateGcode(rvec, conf['graph'], conf['plotter'])

    with open(conf["plotter"]["output"], "w") as stream:
        try:
            stream.write(gcode)
        except:
            exit(1)

def calcMatrix(f, gconf):
    mat = np.ndarray((gconf['density']['x'], gconf['density']['y']))
    dx = (gconf['limits']['xmax'] - gconf['limits']['xmin']) / gconf['density']['x']
    dy = (gconf['limits']['ymax'] - gconf['limits']['ymin']) / gconf['density']['y']
    # TODO: There are more effecient ways to do this with NumPy
    ix = 0
    for x in np.arange(gconf['limits']['xmin'], gconf['limits']['xmax'], dx):
        iy = 0
        for y in np.arange(gconf['limits']['ymin'], gconf['limits']['ymax'], dy):
            mat[ix,iy] = f(x,y)
            iy=iy+1
        ix=ix+1
    
    return mat

def generateVectors(mat, gconf):
    # TODO: Z scaling
    vec = np.ndarray((gconf['density']['x'], gconf['density']['y'], 3))
    dx = (gconf['limits']['xmax'] - gconf['limits']['xmin']) / gconf['density']['x']
    dy = (gconf['limits']['ymax'] - gconf['limits']['ymin']) / gconf['density']['y']
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            vec[x,y] = np.array((gconf['limits']['xmin'] + (x * dx),
                                 gconf['limits']['ymin'] + (y * dy),
                                 mat[x,y] * gconf['limits']['zscale']))
    return vec

def rotateVectors(vec, gconf):
    rvec = np.ndarray((gconf['density']['x'], gconf['density']['y'], 3))
    rx = np.deg2rad(gconf['rotation']['x'])
    ry = np.deg2rad(gconf['rotation']['y'])
    rz = np.deg2rad(gconf['rotation']['z'])
    rotz = np.array([[ np.cos(rz), -np.sin(rz),  0         ],
                     [ np.sin(rz),  np.cos(rz),  0         ],
                     [ 0,           0,           1         ]])
    roty = np.array([[ np.cos(ry),  0,           np.sin(ry)],
                     [ 0,           1,           0         ],
                     [-np.sin(ry),  0,           np.cos(ry)]])
    rotx = np.array([[ 1,           0,           0         ],
                     [ 0,           np.cos(rx), -np.sin(rx)],
                     [ 0,           np.sin(rx),  np.cos(rx)]])
    rot = np.matmul(np.matmul(rotz, roty), rotx)
    for x in range(vec.shape[0]):
        for y in range(vec.shape[1]):
            # TODO: Subtract offset so we are rotating about the correct point (the center of the graph)
            tmp = np.matmul(rotz, vec[x,y])
            tmp = np.matmul(roty, tmp.T)
            
            rvec[x,y] = np.matmul(rotx, tmp.T)
    return rvec

def gcodeDrawto(x, y, pconf):
    # TODO: Account for pen offset
    return "G1 F%f X%f Y%f\n"%(pconf['speed']['draw'],x,y)

def gcodeMoveto(x, y,  pconf):
    # TODO: Account for pen offset
    return "G1 F%f X%f Y%f\n"%(pconf['speed']['travel'],x,y)

def gcodePlunge(pconf):
    return "G1 F%f Z%f\n"%(pconf['speed']['plunge'], pconf['pen']['plunge'])

def gcodeLift(pconf):
    return "G1 F%f Z%f\n"%(pconf['speed']['plunge'], pconf['pen']['lift'])

def generateGcode(vec, gconf, pconf):
    # Single scale to make sure plot is square
    scale = min((gconf["size"]["width"]  / np.max(np.abs(vec[:,:,0]))),
                (gconf["size"]["height"] / np.max(np.abs(vec[:,:,1]))))

    svec = vec * scale

    gcode = ""

    dir = 1

    for x in range(svec.shape[0]):
        if dir == 1:
            gcode += gcodeMoveto(svec[x,0,0], svec[x,0,1], pconf)
        else:
            gcode += gcodeMoveto(svec[x,svec.shape[1]-1,0], svec[x,svec.shape[1]-1,1], pconf)
        gcode += gcodePlunge(pconf)
        for y in range(svec.shape[1]-1):
            if dir == 1:
                gcode += gcodeDrawto(svec[x,y+1,0], svec[x,y+1,1], pconf)
            else:
                gcode += gcodeDrawto(svec[x,svec.shape[1] - (y+2),0], svec[x,svec.shape[1] - (y+2),1], pconf)
        gcode += gcodeLift(pconf)
        dir = -dir

    dir = 1
    
    for y in range(svec.shape[1]):
        if dir == 1:
            gcode += gcodeMoveto(svec[0,y,0], svec[0,y,1], pconf)
        else:
            gcode += gcodeMoveto(svec[svec.shape[0]-1,y,0], svec[svec.shape[0]-1,y,1], pconf)
        gcode += gcodePlunge(pconf)
        for x in range(svec.shape[0]-1):
            if dir == 1:
                gcode += gcodeDrawto(svec[x+1,y,0], svec[x+1,y,1], pconf)
            else:
                gcode += gcodeDrawto(svec[svec.shape[0] - (x+2),y,0], svec[svec.shape[0] - (x+2),y,1], pconf)
        gcode += gcodeLift(pconf)
        dir = -dir
    
    return gcode


# Example functions:

def sinc(x):
    return np.sin(x) / x

def testFunc(x, y):
    return sinc(x) * np.sin(y)

if __name__=="__main__":
    main()