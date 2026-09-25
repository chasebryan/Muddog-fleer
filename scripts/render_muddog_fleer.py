"""Render Muddog-fleer: a procedural, fictional forest FPS film.

The three light pulses follow drawn Bezier curves. Blue/red is a scripted
allegiance state, never an inference about a person or a thermal signature.
No assets, network calls, model weights, or game engine are required.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SCENE = json.loads((ROOT / 'src/muddog/scene.json').read_text())
W, H = SCENE['width'], SCENE['height']
FPS, DURATION = SCENE['fps'], SCENE['duration']
FOCAL = 1200.0
BLUE = np.array([47, 157, 255], dtype=float)
RED = np.array([255, 66, 64], dtype=float)
WHITE = np.array([202, 238, 232], dtype=float)
RNG = np.random.default_rng(SCENE['seed'])


def smooth(a, b, x):
    v = np.clip((x - a) / (b - a), 0, 1)
    return v * v * (3 - 2 * v)


def rgb(value):
    return tuple(np.clip(value, 0, 255).astype(np.uint8))


def camera(t):
    # A continuous, forward-only walk. Pauses and head turns are eased.
    if t < 12:
        z = .60 * t
    elif t < 16:
        q = (t - 12) / 4
        z = 7.2 + 4 * (.60 * q - .52 * (q**3 - .5 * q**4))
    elif t < 26:
        z = 8.56 + .08 * (t - 16)
    elif t < 30:
        q = (t - 26) / 4
        z = 9.36 + 4 * (.08 * q + .52 * (q**3 - .5 * q**4))
    else:
        z = 10.72 + .60 * (t - 30)
    x = .13 * math.sin(t * .24)
    slow = 1 - .86 * smooth(11, 16, t) * (1 - smooth(26, 30, t))
    y = 1.73 + .021 * math.sin(t * 8.7) * slow
    yaw = .037 * math.sin(t * .31)
    yaw -= .21 * smooth(3, 6, t) * (1 - smooth(8, 11, t))
    yaw += .15 * smooth(30, 34, t) * (1 - smooth(36, 39, t))
    lock = float(smooth(11, 16, t) * (1 - smooth(25.5, 29, t)))
    target_yaw = math.atan2(.5 - x, 15.5 - z)
    target_pitch = math.atan2(1.30 - y, math.hypot(.5 - x, 15.5 - z))
    yaw = yaw * (1 - lock) + target_yaw * lock
    pitch = -.035 * (1 - lock) + target_pitch * lock
    return np.array([x, y, z]), yaw, pitch


def subject_state(subject, t):
    if t >= subject.get('goneAt', math.inf):
        return 'diffused'
    if t >= subject.get('diffuseAt', math.inf):
        return 'diffusing'
    if t >= subject.get('redAt', math.inf):
        return 'red'
    return 'blue'


def validate_scene():
    subjects = {s['id']: s for s in SCENE['subjects']}
    assert len(SCENE['projectiles']) == 3, 'Exactly three pulses are required'
    for p in SCENE['projectiles']:
        s = subjects[p['subject']]
        assert subject_state(s, 0) == 'blue'
        assert subject_state(s, p['launch']) == 'red', 'Never launch at blue'
        assert p['launch'] < p['arrival'] < s['diffuseAt']
    assert all(subject_state(subjects[i], DURATION) == 'blue' for i in ['A1', 'A2'])
    samples = np.array([camera(t)[0] for t in np.arange(0, DURATION, 1/FPS)])
    assert np.min(np.diff(samples[:, 2])) >= 0, 'Forward-only camera'
    assert np.max(np.linalg.norm(np.diff(samples, axis=0), axis=1)) < .04, 'No camera jumps'
    for boundary in [12, 16, 26, 30]:
        assert np.linalg.norm(camera(boundary + 1e-6)[0] - camera(boundary - 1e-6)[0]) < 1e-4
    return {'projectiles': 3, 'blue_subjects_preserved': 2,
            'camera_continuous': True, 'classification': 'scripted'}


# World-space geometry: faceted trunks, suspended energy panels and roots.
GEOMETRY = []


def geom(points, color, kind='polygon', width=1, luminous=False):
    GEOMETRY.append((np.array(points, dtype=float), np.array(color, dtype=float), kind, width, luminous))


for _ in range(250):
    z = RNG.uniform(-9, 88)
    x = RNG.uniform(-23, 23)
    if abs(x) < 2.4 + .25 * math.sin(z * .4):
        continue
    height = RNG.uniform(7, 16)
    radius = RNG.uniform(.16, .48)
    lean = RNG.uniform(-.8, .8)
    heat = RNG.uniform(.6, 1.5)
    centers = [(x, 0, z), (x + lean*.18, height*.35, z+.1),
               (x + lean*.65, height*.72, z-.12), (x+lean, height, z)]
    widths = [radius*1.8, radius*.85, radius*.48, radius*.06]
    # Three faces give bark depth and an asymmetrical white-hot edge.
    for i in range(3):
        a, b = np.array(centers[i]), np.array(centers[i+1])
        ra, rb = widths[i], widths[i+1]
        for left, right, brightness in [(-1, -.2, 18), (-.2, .62, 42), (.62, 1, 68)]:
            geom([a+[left*ra,0,0], a+[right*ra,0,0], b+[right*rb,0,0], b+[left*rb,0,0]],
                 np.array([.81,.94,1.0])*brightness*heat)
        geom([a+[-ra*.22,0,-.006], b+[-rb*.22,0,-.006]],
             np.array([98,126,131])*heat, 'line', 1, True)
    # Angular energy bills/panels hang from a branching skeleton.
    for k in range(3):
        by = height * RNG.uniform(.35, .83)
        side = -1 if k % 2 else 1
        length = RNG.uniform(1.3, 3.6)
        a = np.array([x+lean*by/height, by, z])
        b = a + [side*length*.6, .5, RNG.uniform(-.4,.4)]
        c = a + [side*length, RNG.uniform(.7,1.7), RNG.uniform(-.6,.6)]
        geom([a,b,c], [55,76,78], 'line', 2)
        geom([b,c], [80,116,116], 'line', 1, True)
        panel = [b, c, c+[side*.14,-1.3,.2], b+[-side*.15,-.55,.2]]
        geom(panel, [14,31,33])
        geom([panel[0],panel[1],panel[2]], [40,80,80], 'line', 1, True)
        mid = (b+c)/2
        geom([mid,mid+[0,-RNG.uniform(.9,2.5),0]], [48,81,83], 'line', 1)
    for k in range(3):
        a = np.array([x, .025, z])
        b = a + [RNG.uniform(-1.5,1.5), .02, RNG.uniform(-1.2,1.2)]
        geom([a,(a+b)/2+[0,.09,0],b], [45,67,65], 'line', 1, True)

# Floor etchings, fern-like shards, and scattered debris stay anchored in 3D.
for _ in range(400):
    x,z = RNG.uniform(-16,16),RNG.uniform(-5,75)
    if abs(x)<1.6:
        if RNG.random()<.75:
            continue
        h=.035
    else:
        h=RNG.uniform(.1,.55)
    r=RNG.uniform(.05,.22)
    geom([(x-r,0,z),(x,h,z+.04),(x+r,0,z+.1)], [32,53,51])
    if h>.2:
        geom([(x-r,0,z),(x,h,z+.04)], [67,97,83], 'line', 1, True)

MOTES = np.column_stack((RNG.uniform(-13,13,150), RNG.uniform(.15,9,150), RNG.uniform(-3,65,150)))
FRAGMENTS = RNG.normal(size=(120,4))
yy,xx = np.mgrid[0:H,0:W].astype(np.float32)
vignette = np.clip(1 - .53*((xx-W/2)/(W*.67))**2 - .33*((yy-H/2)/(H*.77))**2, .24, 1)
gradient = np.exp(-((yy/H-.44)/.29)**2)
noise = RNG.normal(0, 1.35, (H,W)).astype(np.float32)
BACKGROUND = np.stack([5+gradient*12+noise, 9+gradient*18+noise, 12+gradient*21+noise], axis=2)
BACKGROUND = Image.fromarray(np.uint8(np.clip(BACKGROUND,0,255)))
MONO = os.environ.get('DOG1_MONO_FONT','/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')
SANS = os.environ.get('DOG1_FONT','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
TITLE_FONT = ImageFont.truetype(SANS, 63)
SMALL_FONT = ImageFont.truetype(MONO, 17)


class Frame:
    def __init__(self, t):
        self.t = t
        self.cam,self.yaw,self.pitch = camera(t)
        cy,sy = math.cos(self.yaw),math.sin(self.yaw)
        cp,sp = math.cos(self.pitch),math.sin(self.pitch)
        self.matrix = np.array([[cy,0,-sy],[-sy*sp,cp,-cy*sp],[sy*cp,sp,cy*cp]])
        self.image = BACKGROUND.copy()
        self.d = ImageDraw.Draw(self.image)
        self.light = Image.new('RGB',(W,H))
        self.g = ImageDraw.Draw(self.light)

    def project(self, points):
        q = (np.asarray(points)-self.cam) @ self.matrix.T
        if np.any(q[:,2]<.18):
            return None, float(np.mean(q[:,2]))
        p = q[:,:2] / q[:,2,None] * [FOCAL,-FOCAL] + [W/2,H/2]
        if np.max(p[:,0])< -200 or np.min(p[:,0])>W+200 or np.max(p[:,1])< -400 or np.min(p[:,1])>H+400:
            return None,float(np.mean(q[:,2]))
        return [tuple(v) for v in p],float(np.mean(q[:,2]))

    def line(self, pts, col, width=1, glow=0):
        if len(pts)<2:
            return
        self.d.line(pts,fill=rgb(col),width=width,joint='curve')
        if glow:
            self.g.line(pts,fill=rgb(np.asarray(col)*glow),width=max(2,width+2),joint='curve')

    def world_line(self, pts, col, width=1, glow=0):
        p,z = self.project(pts)
        if p:
            fog=math.exp(-max(0,z)*.026)
            self.line(p,np.asarray(col)*fog,width,glow)

    def disk(self, p, r, col, glow=0):
        box=(*tuple(np.asarray(p)-r),*tuple(np.asarray(p)+r))
        self.d.ellipse(box,fill=rgb(col))
        if glow:
            self.g.ellipse(box,fill=rgb(np.asarray(col)*glow))

    def ground(self):
        # Wide perspective contour ribbons, not a flat screen overlay.
        for z in np.arange(math.floor(self.cam[2]/2)*2, self.cam[2]+65, 2):
            pts=[(x,.009+.008*math.sin(x*2),z+.2*math.sin(x*.8)) for x in np.linspace(-15,15,55)]
            self.world_line(pts,[20,38,39],1)
        for x in [-1.9,1.9]:
            pts=[(x+.12*math.sin(z*.5),.012,z) for z in np.arange(self.cam[2]+.4,self.cam[2]+65,.4)]
            self.world_line(pts,[56,87,82],1,.24)
        # A small number of anchored light shafts reveal forest depth.
        for x,z in [(-6,13),(5,31),(-8,45),(9,56)]:
            p,depth=self.project([(x,16,z),(x-3,0,z-3),(x+.6,0,z-3)])
            if p:
                self.g.polygon(p,fill=rgb(np.array([9,19,18])*math.exp(-max(depth,0)*.023)))

    def draw_geometry(self, item, points, depth):
        _,color,kind,width,luminous=item
        fog=math.exp(-max(0,depth)*.026)
        c=color*fog+np.array([9,17,21])*(1-fog)
        if kind=='polygon':
            self.d.polygon(points,fill=rgb(c))
        else:
            self.line(points,c,width,.34 if luminous else 0)

    def mannequin(self, subject):
        t=self.t
        state=subject_state(subject,t)
        if state=='diffused':
            return
        pos=np.array(subject['position'],dtype=float)
        pos[1]=.025*math.sin(t*1.6+pos[2])
        col=RED if state in ('red','diffusing') else BLUE
        diffusion=float(smooth(subject.get('diffuseAt',100),subject.get('goneAt',103),t))
        visibility=1-diffusion
        # Faceless light mannequin. Its anatomy is drawn geometry, not imagery.
        sway=.026*math.sin(t*1.9+pos[2])
        joints={'head':(sway,2.04,0),'neck':(sway,1.82,0),'hip':(0,.99,0),
                'ls':(-.29+sway,1.70,0),'rs':(.29+sway,1.70,0),
                'le':(-.39,1.37,.01),'re':(.38,1.37,.04),
                'lh':(-.35,1.05,-.04),'rh':(.37,1.07,-.03),
                'lk':(-.18,.55,-.02),'rk':(.20,.55,.08),
                'lf':(-.22,.06,-.11),'rf':(.23,.06,-.11)}
        P={k:pos+v for k,v in joints.items()}
        center,depth=self.project([pos+[0,1.3,0]])
        if not center:
            return
        scale=FOCAL/depth
        parts=[('hip','lk',.115),('hip','rk',.115),('lk','lf',.09),('rk','rf',.09),
               ('ls','le',.092),('le','lh',.072),('rs','re',.092),('re','rh',.072)]
        if visibility>.035:
            for a,b,width in parts:
                p,_=self.project([P[a],P[b]])
                if p:
                    self.line(p,col*.59*visibility,max(2,int(scale*width*2)),.60)
                    self.line(p,col*.95*visibility,max(1,int(scale*width*.60)),.60)
            torso=[pos+v for v in [(-.28+sway,1.72,0),(.28+sway,1.72,0),(.23,1.21,0),(.15,.94,0),(-.15,.94,0),(-.23,1.21,0)]]
            p,_=self.project(torso)
            if p:
                self.d.polygon(p,fill=rgb(col*.28*visibility))
                self.line(p+[p[0]],col*.85*visibility,2,.70)
            head,_=self.project([P['head']])
            if head:
                x,y=head[0];rx,ry=scale*.145,scale*.18
                b=(x-rx,y-ry,x+rx,y+ry)
                self.d.ellipse(b,fill=rgb(col*.40*visibility),outline=rgb(col*visibility),width=2)
                self.g.ellipse(b,outline=rgb(col*.7*visibility),width=3)
            # Luminous ribs and spine make a synthetic, layered silhouette.
            for h in np.linspace(1.16,1.67,9):
                w=.21-.02*math.sin(h*7)
                self.world_line([pos+[-w,h,0],pos+[0,h-.035,-.025],pos+[w,h,0]],col*.7*visibility,1,.6)
            self.world_line([pos+[0,1.77,-.03],pos+[0,1.0,-.03]],col*visibility,2,.8)
            for k in ['ls','rs','le','re','hip','lk','rk']:
                p,_=self.project([P[k]])
                if p:self.disk(p[0],max(2,scale*.025),WHITE*.85*visibility,.7)
        if subject['id']=='T1' and t>11:
            # World-space beacon ellipses react to proximity and film time.
            amp=(1-diffusion)*float(smooth(10,16,t))
            for j in range(3):
                phase=((t*(.60 if t<19 else .95)+j/3)%1)
                radius=.48+phase*.8
                pts=[pos+[radius*math.cos(a),.055,radius*math.sin(a)] for a in np.linspace(0,math.tau,80)]
                self.world_line(pts,col*amp*(1-phase)*.55,1,.65)
            # Short curves orbit the torso; they are environmental traces.
            for side in [-1,1]:
                pts=[pos+[side*(.38+.44*q),1.30+.18*math.sin(q*6+t*2),-.30-q*.55] for q in np.linspace(0,1,35)]
                self.world_line(pts,col*.65*amp,1,.9)
        if state=='diffusing':
            age=t-subject['diffuseAt']
            for i,v in enumerate(FRAGMENTS):
                origin=pos+np.array([v[0]*.17,.2+(i%21)/11,.03])
                offset=np.array([v[1]*age*.45,age*(.8+abs(v[2])*.22),v[3]*age*.24])
                p,z=self.project([origin+offset,origin+offset+[0,.04+.06*age,0]])
                if p:self.line(p,(RED*(1-diffusion)+WHITE*diffusion)*(1-diffusion),2,.8)
            radius=age*1.4
            pts=[pos+[radius*math.cos(a),.02,radius*math.sin(a)] for a in np.linspace(0,math.tau,120)]
            self.world_line(pts,WHITE*(1-diffusion),2,.7)

    def particles(self):
        for i,m in enumerate(MOTES):
            p,z=self.project([m+[.035*math.sin(self.t+i),.07*math.sin(self.t*.3+i),0]])
            if p:
                flicker=.45+.25*math.sin(self.t*.9+i*3)
                self.disk(p[0],1 if z>8 else 1.5,np.array([97,133,128])*flicker*math.exp(-z*.028),.4)

    def pulses(self):
        t=self.t
        target=np.array(SCENE['subjects'][1]['position'])+[0,1.3,0]
        end,depth=self.project([target])
        if not end:return
        ex,ey=end[0]
        for event in SCENE['projectiles']:
            start,arrival=event['launch'],event['arrival']
            if start<=t<=arrival:
                q=(t-start)/(arrival-start)
                port=event['port']
                p0=np.array([W/2+port*W*.29,H*1.04])
                p1=np.array([W/2+port*W*.38,H*.61])
                p2=np.array([ex-port*95,ey-75])
                p3=np.array([ex,ey])
                def bezier(s):return (1-s)**3*p0+3*(1-s)**2*s*p1+3*(1-s)*s*s*p2+s**3*p3
                tail=max(0,q-.19)
                pts=[tuple(bezier(s)) for s in np.linspace(tail,q,40)]
                self.line(pts,[56,176,239],8,.85)
                self.line(pts,[201,245,255],3,1)
                head=bezier(q)
                self.disk(head,6,[230,255,251],1)
                self.disk(head,2,[255,255,255],1)
            age=t-arrival
            if 0<=age<.50:
                fade=(1-age/.50)
                r=14+age*160
                for j in range(2):
                    pts=[(ex+math.cos(a)*(r+j*11),ey+math.sin(a)*(r+j*11)*.73) for a in np.linspace(0,math.tau,80)]
                    self.line(pts,WHITE*fade,2,.9)
                for a in np.linspace(0,math.tau,9)[:-1]:
                    self.line([(ex+math.cos(a)*r*.3,ey+math.sin(a)*r*.3),(ex+math.cos(a)*r,ey+math.sin(a)*r)],WHITE*fade,1,.6)

    def reticle(self):
        t=self.t
        entrance=float(smooth(1.8,3,t))
        fade=1-float(smooth(38.5,40,t))
        lock=float(smooth(12,18,t))*(1-float(smooth(23.5,27,t)))
        red=float(smooth(18.96,19.12,t))*(1-float(smooth(24,26,t)))
        col=(WHITE*(1-lock)+BLUE*lock)*(1-red)+RED*red
        col*=entrance*fade
        radius=35-7*lock+1.3*math.sin(t*2.4)
        x,y=W/2,H/2
        # A single reticle formed from open hyperbolic arcs. No other HUD.
        for angle in [0,math.pi/2,math.pi,math.pi*1.5]:
            pts=[]
            for u in np.linspace(-.76,.76,28):
                a=radius+9*(math.cosh(u*1.8)-1)
                b=12*math.sinh(u)
                pts.append((x+a*math.cos(angle)-b*math.sin(angle),y+a*math.sin(angle)+b*math.cos(angle)))
            self.d.line(pts,fill=(3,10,15),width=6,joint='curve')
            self.line(pts,col,2,.8)
        self.disk((x,y),1.5,col,.8)
        if 12<t<26:
            breathe=(math.sin(t*4.5)+1)/2
            for offset in [0,math.pi]:
                points=[(x+(radius+10)*math.cos(a),y+(radius+10)*math.sin(a)) for a in np.linspace(offset+t*.6,offset+t*.6+.6+lock*.8,24)]
                self.line(points,col*(.28+breathe*.5),1,.5)

    def finish(self):
        # Thermal-style halation, a soft vignette, and fine fixed-pattern noise.
        bloom=self.light.resize((W//4,H//4),Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(5))
        bloom=bloom.resize((W,H),Image.Resampling.BILINEAR)
        im=ImageChops.add(self.image,bloom,scale=1)
        arr=np.asarray(im).astype(np.float32)
        arr*=vignette[:,:,None]
        # Restrained sensor scan rows, with no full-frame flashes or strobes.
        arr[::4]*=.974
        im=Image.fromarray(np.uint8(np.clip(arr,0,255)))
        title=float(smooth(.2,1.0,self.t))*(1-float(smooth(1.7,2.8,self.t)))
        title=max(title,float(smooth(39,40.3,self.t)))
        if title>0:
            overlay=Image.new('RGBA',(W,H))
            d=ImageDraw.Draw(overlay)
            d.rectangle((0,0,W,H),fill=(0,5,9,int(100*title)))
            d.text((W/2,H*.43),'MUDDOG-FLEER',font=TITLE_FONT,fill=(219,241,238,int(255*title)),anchor='mm',stroke_width=0)
            d.text((W/2,H*.50),'D O G 1   /   S Y N T H E T I C   F O R E S T',font=SMALL_FONT,fill=(126,175,179,int(255*title)),anchor='mm')
            im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB')
        return im


def draw_frame(t):
    f=Frame(t)
    f.ground()
    visible=[]
    for item in GEOMETRY:
        p,depth=f.project(item[0])
        if p:visible.append((depth,'geometry',item,p))
    for subject in SCENE['subjects']:
        p,depth=f.project([np.array(subject['position'])+[0,1.3,0]])
        if p:visible.append((depth,'subject',subject,None))
    for depth,kind,item,points in sorted(visible,key=lambda v:v[0],reverse=True):
        if kind=='geometry':f.draw_geometry(item,points,depth)
        else:f.mannequin(item)
    f.particles()
    f.pulses()
    f.reticle()
    return f.finish()


def synth_audio(path):
    sr=48000
    t=np.arange(sr*DURATION,dtype=np.float64)/sr
    rng=np.random.default_rng(SCENE['seed']+1)
    out=np.zeros((len(t),2),np.float64)
    # Organic pump + robotic respiration: deliberately synthetic, non-vocal.
    pad=.018*np.sin(2*np.pi*47*t+.30*np.sin(t*.41))+.011*np.sin(2*np.pi*71*t)
    air=rng.normal(0,1,len(t))
    air=np.convolve(air,np.ones(28)/28,mode='same')
    breath=(.5+.5*np.sin(t*math.tau/4.8))**3
    bed=pad+air*(.025+.065*breath)
    out[:,0]=bed*(.89+.05*np.sin(t*.6))
    out[:,1]=bed*(.89+.05*np.cos(t*.6))
    cues=[]

    def add(at,signal,pan=0,kind=None):
        start=int(at*sr)
        count=min(len(signal),len(out)-start)
        if count<=0:return
        pan=float(np.clip(pan,-1,1))
        out[start:start+count,0]+=signal[:count]*math.sqrt((1-pan)/2)
        out[start:start+count,1]+=signal[:count]*math.sqrt((1+pan)/2)
        if kind:cues.append({'time':round(at,4),'type':kind,'pan':round(pan,3)})

    def pulse(duration,f0,f1,amp,texture=.0):
        q=np.arange(int(duration*sr))/sr
        env=np.sin(np.pi*q/duration)**2
        phase=math.tau*(f0*q+(f1-f0)*q*q/(2*duration))
        return amp*env*(np.sin(phase)+.24*np.sin(phase*1.503)+texture*rng.normal(0,1,len(q)))

    for at in np.arange(.9,39,.72):
        if 14<at<27:continue
        q=np.arange(int(.22*sr))/sr
        step=.16*np.exp(-q*25)*np.sin(math.tau*(62*q-65*q*q))
        step+=.018*rng.normal(0,1,len(q))*np.exp(-q*32)
        add(at,step,(-1 if int(at/.72)%2 else 1)*.33,'step')
    for at in np.arange(1,39,1.25):
        add(at,pulse(.14,82,46,.07),-.08,'pump')
        add(at+.19,pulse(.11,67,40,.045),.08)
    at=10.0
    while at<23:
        proximity=float(smooth(10,19,at))
        interval=.95-.66*proximity
        pos,yaw,_=camera(at)
        angle=math.atan2(.5-pos[0],15.5-pos[2])-yaw
        add(at,pulse(.095,630+proximity*590,830+proximity*490,.105),np.clip(angle*2,-.7,.7),'beacon')
        at+=interval
    add(19,pulse(.48,280,690,.14,.015),0,'red-state')
    for e in SCENE['projectiles']:
        add(e['launch'],pulse(.36,160,1850,.30,.045),e['port']*.65,e['id']+'-launch')
        add(e['arrival'],pulse(.26,920,180,.24,.08),0,e['id']+'-arrival')
        add(e['arrival']+.09,pulse(.30,700,410,.065),-e['port']*.35)
    add(23,pulse(1.9,430,110,.15,.07),0,'diffuse')
    for at in np.arange(23.1,25.2,.17):
        add(at,pulse(.22,1800-(at-23)*430,1250,.035),math.sin(at*8)*.7)
    add(31.5,pulse(.16,630,780,.07),.6,'blue-observe')
    add(33,pulse(.16,630,780,.06),.6,'blue-observe')
    fade=np.minimum(np.clip(t/1.2,0,1),np.clip((DURATION-t)/2.2,0,1))
    out*=fade[:,None]
    peak=float(np.max(np.abs(out)))
    out*=.79/max(peak,1e-9)
    pcm=np.round(out*32767).astype('<i2')
    with wave.open(str(path),'wb') as w:
        w.setnchannels(2);w.setsampwidth(2);w.setframerate(sr);w.writeframes(pcm.tobytes())
    return {'sampleRate':sr,'channels':2,'peak':float(np.max(np.abs(out))),
            'rms':float(np.sqrt(np.mean(out*out))),'cues':cues}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work-dir',type=Path)
    parser.add_argument('--preview',action='store_true',help='Render a contact sheet and scene checks only')
    parser.add_argument('--output',type=Path,default=ROOT/'media/Muddog-fleer.mp4')
    args=parser.parse_args()
    checks=validate_scene()
    work=args.work_dir or Path(tempfile.mkdtemp(prefix='muddog-fleer-'))
    work.mkdir(parents=True,exist_ok=True)
    reviews=[1,7,16.8,19.5,20.92,21.7,22.4,23.8,32,39.9]
    sheet=Image.new('RGB',(W,H//2*5))
    for i,t in enumerate(reviews):
        im=draw_frame(t)
        im.save(work/f'frame-{t:05.2f}.png')
        tile=im.resize((W//2,H//2),Image.Resampling.LANCZOS)
        ImageDraw.Draw(tile).text((15,15),f'{t:05.2f}s',font=SMALL_FONT,fill=(245,246,245),stroke_width=2,stroke_fill=(0,0,0))
        sheet.paste(tile,((i%2)*W//2,(i//2)*H//2))
    sheet.save(work/'contact-sheet.jpg',quality=93)
    print(json.dumps({'checks':checks,'preview':str(work/'contact-sheet.jpg')}),flush=True)
    if args.preview:return
    audio=synth_audio(work/'audio.wav')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{W}x{H}','-r',str(FPS),'-i','-','-i',str(work/'audio.wav'),
         '-c:v','libx264','-preset','slow','-crf','24','-pix_fmt','yuv420p',
         '-c:a','aac','-b:a','192k','-t',str(DURATION),'-movflags','+faststart',
         '-metadata','title=Muddog-fleer — DOG1 synthetic forest',str(args.output)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for i in range(DURATION*FPS):
            process.stdin.write(draw_frame(i/FPS).tobytes())
            if i%(FPS*2)==0:print(f'Rendered {i//FPS:02d} / {DURATION} seconds',flush=True)
    finally:
        process.stdin.close()
    if process.wait():raise RuntimeError('FFmpeg failed')
    manifest={'format':'Muddog-fleer/video-v1','scene':SCENE,'checks':checks,'audio':audio,
              'video':{'file':args.output.name,'width':W,'height':H,'fps':FPS,'duration':DURATION,
                       'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()},
              'renderer':'Deterministic procedural 3D projection, vector silhouettes, bloom and drawn pulse paths.'}
    args.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(str(args.output),flush=True)
    if not args.work_dir:shutil.rmtree(work)


if __name__=='__main__':
    main()
