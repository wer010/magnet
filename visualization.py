import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_by_go(pos,mag):
    fig = go.Figure()
    fig.add_trace(go.Cone(x=[1, ] * 3, name="base"))
    fig.add_trace(go.Cone(x=[2, ] * 3, opacity=0.3, name="opacity:0.3"))
    fig.add_trace(go.Cone(x=[3, ] * 3, lighting_ambient=0.3, name="lighting.ambient:0.3"))
    fig.add_trace(go.Cone(x=[4, ] * 3, lighting_diffuse=0.3, name="lighting.diffuse:0.3"))
    fig.add_trace(go.Cone(x=[5, ] * 3, lighting_specular=2, name="lighting.specular:2"))
    fig.add_trace(go.Cone(x=[6, ] * 3, lighting_roughness=1, name="lighting.roughness:1"))
    fig.add_trace(go.Cone(x=[7, ] * 3, lighting_fresnel=2, name="lighting.fresnel:2"))
    fig.add_trace(go.Cone(x=[8, ] * 3, lightposition=dict(x=0, y=0, z=1e5),
                          name="lighting.position x:0,y:0,z:1e5"))

    fig.update_traces(y=[1, 2, 3], z=[1, 1, 1],
                      u=[1, 2, 3], v=[1, 1, 2], w=[4, 4, 1],
                      hoverinfo="u+v+w+name",
                      showscale=False)

    fig.update_layout(scene=dict(aspectmode="data",
                                 camera_eye=dict(x=0.05, y=-2.6, z=2)),
                      margin=dict(t=0, b=0, l=0, r=0))

    fig.show()

    return 0

def plot_2d(pos, mag):
    fig, ax = plt.subplots()
    ax.quiver(pos[:,:,0].reshape(-1), pos[:,:,2].reshape(-1),
              mag[:,:,0].reshape(-1), mag[:,:,2].reshape(-1), color='tab:blue')
    ax.scatter(pos[:,:,0].reshape(-1), pos[:,:,2].reshape(-1), s=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Magnetic Field')
    label = ['direction', 'the describe point']
    ax.legend(label, loc='best', shadow='True')
    plt.show()
    return 0


def plot_3d_by_matplotlib(pos,mag):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.quiver(pos[:,:,:,0].reshape(-1), pos[:,:,:,1].reshape(-1), pos[:,:,:,2].reshape(-1),
              mag[:,:,:,0].reshape(-1), mag[:,:,:,1].reshape(-1), mag[:,:,:,2].reshape(-1), length=0.1)

    plt.show()
    return 0