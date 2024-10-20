# main.py

# import libraries
import pygame
import time
import numpy as np
from classes.pygame.pygame3d import Pygame3d


# setup 3d
gl = Pygame3d()
pygame.mouse.set_visible(False)

# add models to scene
model = gl.load_model("./assets/3d/cube.csv")
scene = [model]


f = 0 # last frame number
ftt = 0 # total frame time
fdx = 0 # number of frames
# game loop
while gl.running:
    f+=1
    gl.check_for_quit()

    # input capture
    keys = pygame.key.get_pressed()
    if keys[pygame.K_q]:
        pygame.quit()

    # get x,y of mouse and changes the origin to the top left corner
    x, y = gl.to_rotation_coordinates(*pygame.mouse.get_pos())
    pygame.mouse.set_pos(gl.center)

    # rotate camera based on mouse movement
    gl.camera.rotation[0] += y / 2
    gl.camera.rotation[1] += -x / 2

    # draw a cursor
    pygame.draw.circle(gl.screen, "white", gl.center, 3)

    # calculate the pitch and yaw of the camera
    yaw = np.radians(gl.camera.rotation[1])
    pitch = np.radians(gl.camera.rotation[0])
    
    # calculate the direction the camera is facing
    forward = np.array([
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
        np.cos(pitch) * np.cos(yaw)
    ])
    
    # normalize vector
    forward /= np.linalg.norm(forward)

    if keys[pygame.K_w]: # move the camera forward
        gl.camera.position += forward * 0.1

    if keys[pygame.K_s]:
        gl.camera.position -= forward * 0.1

    if keys[pygame.K_a]:
        gl.camera.position[-1] += -0.1

    if keys[pygame.K_d]:
        gl.camera.position[-1] += 0.1

    # project the scene onto the screen
    print("Projecting frame",f)
    start = time.time()
    projected_mesh = gl.project_scene(scene)
    end = time.time()
    project_latency = end - start
    print("Projected in",project_latency,"s")

    # render the projection
    print("Rendering frame",f)
    start = time.time()
    gl.render_scene(projected_mesh)
    end = time.time()
    render_latency = end - start
    print("Rendered in",render_latency,"s")
    ftt += render_latency + project_latency
    fdx += 1
    

    # rotate the cube model
    scene[0].rotation += 1

    # debug info
    text_surface = gl.font.render(f"Frame: {f} | Avg Frametime: {ftt/fdx} | Rotation: {gl.camera.rotation} | Position: {gl.camera.position} | {project_latency} | {render_latency}", True, (255, 255, 255))
    gl.screen.blit(text_surface, (0, 0))
    gl.flip()