"""Procedural scenery for a fictional thermal-style game film.

Everything here is visual geometry in arbitrary scene units, with no sensor,
weapon, terrain-analysis or tactical model.
"""
import math
import numpy as np


def ground_height(x, z):
    return (.10*np.sin(z*.23) + .09*np.sin(x*.37+z*.17)
            + .025*np.sin(z*.92+x*1.6))


def build_forest(seed):
    rng=np.random.default_rng(seed)
    items=[]
    tint=np.array([.94,.98,1.0])

    def add(points, shade, kind='polygon', width=1):
        items.append((np.asarray(points,dtype=np.float32),tint*shade,kind,width,False))

    def tube(a,b,r0,r1,shade,sides=6):
        a,b=np.asarray(a),np.asarray(b)
        direction=b-a;direction/=np.linalg.norm(direction)
        basis=np.cross(direction,[0,1,0])
        if np.linalg.norm(basis)<.01:basis=np.cross(direction,[1,0,0])
        basis/=np.linalg.norm(basis);other=np.cross(direction,basis)
        ring=[basis*math.cos(q)+other*math.sin(q) for q in np.linspace(0,math.tau,sides,endpoint=False)]
        for j in range(sides):
            u,v=ring[j],ring[(j+1)%sides]
            light=.62+.31*abs(np.dot((u+v)/2,[-.6,.5,-.6]))
            add([a+u*r0,a+v*r0,b+v*r1,b+u*r1],shade*light)

    # Ground is shaded as a continuous height field by the film renderer.
    for _ in range(220):
        x,z=rng.uniform(-22,22),rng.uniform(-6,83)
        if abs(x)<2.25+.18*math.sin(z*.5):continue
        y=ground_height(x,z)
        height=rng.uniform(7.5,14)
        radius=rng.uniform(.13,.38)
        lean=rng.normal(0,.25,2)
        rings=[np.array([x+lean[0]*q,y+height*q,z+lean[1]*q]) for q in [0,.12,.42,.75,1]]
        radii=[radius*1.4,radius,radius*.72,radius*.42,.025]
        heat=rng.uniform(49,93)
        for a,b,ra,rb in zip(rings,rings[1:],radii,radii[1:]):tube(a,b,ra,rb,heat)
        # Bark furrows run around the trunk rather than forming flat panels.
        for j in range(8):
            angle=j*math.tau/8
            pts=[]
            for k,q in enumerate(np.linspace(0,.85,8)):
                r=radius*(1-.67*q)
                pts.append((x+lean[0]*q+r*math.cos(angle)+.012*math.sin(k*2+j),
                            y+height*q,z+lean[1]*q+r*math.sin(angle)))
            add(pts,heat*(.38 if j%2 else .78),'line',1)
        for j in range(5):
            a=j*math.tau/5+rng.uniform(-.3,.3)
            end=np.array([x+math.cos(a)*radius*4,0,z+math.sin(a)*radius*4])
            end[1]=ground_height(end[0],end[2])+.035
            tube([x,y+.32,z],end,radius*.35,.018,heat*.65,5)
        # Branching conifer limbs and irregular needle clusters.
        for j in range(7):
            h=height*(.34+j*.078)
            angle=j*2.4+rng.uniform(-.6,.6)
            length=rng.uniform(1.5,3.4)*(1-j*.055)
            a=np.array([x+lean[0]*h/height,y+h,z+lean[1]*h/height])
            vec=np.array([math.cos(angle),.17,math.sin(angle)])
            b=a+vec*length
            tube(a,b,radius*.23,.018,heat*.78,5)
            for k in range(2):
                center=a+vec*length*(.56+k*.32)
                side=np.array([-math.sin(angle),0,math.cos(angle)])
                span=length*(.36-.12*k)
                points=[]
                for q in np.linspace(-1,1,9):
                    points.append(center+side*q*span+vec*(.16 if len(points)%2 else -.16)+[0,-abs(q)*.23,0])
                points += [center+vec*.48+[0,.10,0]]
                add(points,rng.uniform(19,36))
                add([points[0],center+[0,.02,0],points[8]],rng.uniform(40,61),'line',1)

    # Fern fronds and fine grasses create depth and partial foreground occlusion.
    for _ in range(230):
        x,z=rng.uniform(-15,15),rng.uniform(-4,65)
        if abs(x)<1.45:continue
        base=np.array([x,ground_height(x,z),z])
        height=rng.uniform(.22,.80)
        for j in range(4):
            angle=j*math.pi/2+rng.uniform(-.5,.5)
            outward=np.array([math.cos(angle),0,math.sin(angle)])
            side=np.array([-math.sin(angle),0,math.cos(angle)])
            pts=[base+outward*(q*.62)+[0,height*math.sin(q*math.pi*.68),0] for q in np.linspace(0,1,5)]
            add(pts,rng.uniform(48,69),'line',1)
            for k,q in enumerate([.30,.47,.65,.82]):
                c=base+outward*(q*.62)+[0,height*math.sin(q*math.pi*.68),0]
                for sign in [-1,1]:
                    tip=c+side*sign*(.18*(1-q)+.035)+outward*.12+[0,-.03,0]
                    add([c,tip,c+outward*.07+[0,.035,0]],rng.uniform(32,60))

    for _ in range(180):
        x,z=rng.uniform(-13,13),rng.uniform(-4,70)
        base=np.array([x,ground_height(x,z),z])
        radius=rng.uniform(.05,.24)
        height=radius*rng.uniform(.35,.8)
        ring=[base+[math.cos(a)*radius,0,math.sin(a)*radius] for a in np.linspace(0,math.tau,6,endpoint=False)]
        top=base+[radius*.13,height,radius*.12]
        for j in range(6):add([ring[j],ring[(j+1)%6],top],rng.uniform(30,56))
    for _ in range(25):
        x,z=rng.uniform(-10,10),rng.uniform(1,70)
        if abs(x)<2:continue
        a=np.array([x,ground_height(x,z)+.08,z])
        b=a+[rng.uniform(-1.5,1.5),0,rng.uniform(.5,2)]
        b[1]=ground_height(b[0],b[2])+.07
        tube(a,b,.085,.03,54,5)
    return items


class ProjectionBatch:
    """Project a static mesh in one array operation per frame."""
    def __init__(self,geometry):
        lengths=np.array([len(g[0]) for g in geometry])
        self.starts=np.r_[0,np.cumsum(lengths)[:-1]]
        self.ends=self.starts+lengths
        self.vertices=np.concatenate([g[0] for g in geometry])

    def project(self,camera,matrix,focal,width,height):
        v=self.vertices-camera
        q=v@matrix.T
        depth=np.add.reduceat(q[:,2],self.starts)/(self.ends-self.starts)
        near=np.minimum.reduceat(q[:,2],self.starts)>.16
        p=q[:,:2]/np.maximum(q[:,2,None],.05)*[focal,-focal]+[width/2,height/2]
        lo_x=np.minimum.reduceat(p[:,0],self.starts);hi_x=np.maximum.reduceat(p[:,0],self.starts)
        lo_y=np.minimum.reduceat(p[:,1],self.starts);hi_y=np.maximum.reduceat(p[:,1],self.starts)
        visible=near&(hi_x>-20)&(lo_x<width+20)&(hi_y>-20)&(lo_y<height+20)&(depth<85)
        ids=np.flatnonzero(visible)
        output=[(float(depth[i]),int(i),[tuple(v) for v in p[self.starts[i]:self.ends[i]]]) for i in ids]
        # Clip near-plane crossings instead of dropping whole ground triangles.
        crossings=(~near)&(np.maximum.reduceat(q[:,2],self.starts)>.16)
        for i in np.flatnonzero(crossings):
            poly=list(q[self.starts[i]:self.ends[i]])
            clipped=[]
            for a,b in zip(poly,poly[1:]+poly[:1]):
                inside_a,inside_b=a[2]>=.16,b[2]>=.16
                if inside_a:clipped.append(a)
                if inside_a!=inside_b:
                    clipped.append(a+(b-a)*((.16-a[2])/(b[2]-a[2])))
            if len(clipped)<2:continue
            points=np.array(clipped)
            xy=points[:,:2]/points[:,2,None]*[focal,-focal]+[width/2,height/2]
            if xy[:,0].max()<-20 or xy[:,0].min()>width+20 or xy[:,1].max()<-20 or xy[:,1].min()>height+20:continue
            output.append((float(points[:,2].mean()),int(i),[tuple(v) for v in xy]))
        return output
