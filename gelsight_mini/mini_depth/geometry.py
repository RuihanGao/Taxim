"""Depth-only geometry, mm; no optical renderer or deformation model imports."""
import numpy as np
from numba import njit

@njit(cache=True)
def raster(vertices, faces, width, height, perspective=False):
    """Pixel centers at integer coordinates; reciprocal camera Z for perspective."""
    out=np.full((height,width),np.inf if perspective else -np.inf)
    ids=np.full((height,width),-1,np.int32)
    for fi in range(len(faces)):
        p=vertices[faces[fi]]
        den=(p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
        if abs(den)<1e-12:continue
        x0=max(0,int(np.ceil(np.min(p[:,0]))));x1=min(width-1,int(np.floor(np.max(p[:,0]))))
        y0=max(0,int(np.ceil(np.min(p[:,1]))));y1=min(height-1,int(np.floor(np.max(p[:,1]))))
        for y in range(y0,y1+1):
            for x in range(x0,x1+1):
                a=((p[1,1]-p[2,1])*(x-p[2,0])+(p[2,0]-p[1,0])*(y-p[2,1]))/den
                b=((p[2,1]-p[0,1])*(x-p[2,0])+(p[0,0]-p[2,0])*(y-p[2,1]))/den;c=1-a-b
                if min(a,b,c)<-1e-8:continue
                z=1/(a/p[0,2]+b/p[1,2]+c/p[2,2]) if perspective else a*p[0,2]+b*p[1,2]+c*p[2,2]
                if (perspective and z<out[y,x]) or (not perspective and z>out[y,x]):out[y,x]=z;ids[y,x]=fi
    return out,ids

def project(v,p):
    z=p['distance_mm']-v[:,2]
    if np.any(z<=p['near_mm']):raise ValueError('Geometry crosses camera near plane; triangle clipping is required.')
    return np.c_[p['fx']*v[:,0]/z+p['width']/2,-p['fy']*v[:,1]/z+p['height']/2,z]

def camera(v,f,p):return raster(project(v,p),f,p['width'],p['height'],True)

def unproject(depth,p):
    y,x=np.indices(depth.shape)
    return np.stack([(x-p['width']/2)*depth/p['fx'],-(y-p['height']/2)*depth/p['fy'],p['distance_mm']-depth],axis=-1)

def normal_grid(v,f,spacing=.05,extent=(20.75,25.25)):
    w=int(round(extent[0]/spacing))+1;h=int(round(extent[1]/spacing))+1
    return raster(np.c_[v[:,0]/spacing+(w-1)/2,-v[:,1]/spacing+(h-1)/2,v[:,2]],f,w,h)

def pad_frame(pad,center,roll=90):
    corners=np.asarray(pad['corners_xyz'],dtype=float);o=corners.mean(0)
    _,s,vh=np.linalg.svd(corners-o,full_matrices=False);n=vh[-1]
    if s[1]<1e-8:raise ValueError('Degenerate pad.')
    if np.dot(n,pad.get('normal_toward_object',center-o))>0:n=-n
    e=np.roll(corners,-1,axis=0)-corners;u=e[np.argmax(np.linalg.norm(e,axis=1))];u-=n*np.dot(u,n);u/=np.linalg.norm(u)
    v=np.cross(n,u);a=np.deg2rad(roll);b=np.column_stack([u*np.cos(a)+v*np.sin(a),-u*np.sin(a)+v*np.cos(a),n])
    return o,b,(corners-o)@b
