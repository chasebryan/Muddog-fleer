"""Render Muddog-fleer: a procedural, fictional forest FPS film.

The three light pulses follow drawn Bezier curves. Blue/red is a scripted
allegiance state, never an inference about a person or a thermal signature.
No assets, network calls, model weights, or game engine are required.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from muddog.forest import build_forest, ground_height, ProjectionBatch
SCENE = json.loads((ROOT / 'src/muddog/scene.json').read_text())
W, H = SCENE['width'], SCENE['height']
FPS, DURATION = SCENE['fps'], SCENE['duration']
FOCAL = 1320.0
BLUE = np.array([68, 155, 230], dtype=float)
RED = np.array([225, 71, 53], dtype=float)
WHITE = np.array([204, 218, 218], dtype=float)
CHAR_SCALE = .87
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
    y = 1.70 + ground_height(x,z) + .011 * math.sin(t * 8.7) * slow
    yaw = .021 * math.sin(t * .31)
    yaw -= .21 * smooth(3, 6, t) * (1 - smooth(8, 11, t))
    yaw += .15 * smooth(30, 34, t) * (1 - smooth(36, 39, t))
    lock = float(smooth(11, 16, t) * (1 - smooth(25.5, 29, t)))
    target_yaw = math.atan2(.5 - x, 15.5 - z)
    target_pitch = math.atan2(1.30*CHAR_SCALE+ground_height(.5,15.5)-y, math.hypot(.5-x,15.5-z))
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
    tree=ast.parse(Path(__file__).read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and n.func.attr in {'text','multiline_text','textbbox','multiline_textbbox'}
                   for n in ast.walk(tree)), 'No text drawing anywhere in the film renderer'
    return {'projectiles': 3, 'blue_subjects_preserved': 2,
            'camera_continuous': True, 'classification': 'scripted', 'text_rendering_calls': 0}


GEOMETRY = build_forest(SCENE['seed'])
MESH = ProjectionBatch(GEOMETRY)
MOTES = np.column_stack((RNG.uniform(-13,13,50), RNG.uniform(.15,9,50), RNG.uniform(-3,65,50)))
FRAGMENTS = RNG.normal(size=(120,4))
yy,xx = np.mgrid[0:H,0:W].astype(np.float32)
vignette = np.clip(1 - .53*((xx-W/2)/(W*.67))**2 - .33*((yy-H/2)/(H*.77))**2, .24, 1)
gradient = np.exp(-((yy/H-.44)/.29)**2)
noise = RNG.normal(0, .55, (H,W)).astype(np.float32)
BACKGROUND = np.stack([15+gradient*23+noise, 19+gradient*24+noise, 21+gradient*24+noise], axis=2)
BACKGROUND = Image.fromarray(np.uint8(np.clip(BACKGROUND,0,255)))
fy,fx=np.mgrid[0:H//2,0:W//2].astype(np.float32)
FLOOR_RAYS=np.stack(((fx*2-W/2)/FOCAL,-(fy*2-H/2)/FOCAL,np.ones_like(fx)),axis=2)


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
        self.effects = Image.new('RGB',(W,H))
        self.e = ImageDraw.Draw(self.effects)

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

    def effect_line(self,pts,col,width=1):
        # Additive light cannot darken a surface as an effect fades.
        if len(pts)<2:return
        self.e.line(pts,fill=rgb(col),width=width,joint='curve')
        self.g.line(pts,fill=rgb(np.asarray(col)*.65),width=width+1,joint='curve')

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
        # World-anchored shading avoids a tiled or polygonal floor appearance.
        rays=FLOOR_RAYS@self.matrix
        denominator=np.minimum(rays[:,:,1],-.001)
        distance=np.clip((ground_height(self.cam[0],self.cam[2])-self.cam[1])/denominator,0,90)
        for _ in range(2):
            wx=self.cam[0]+rays[:,:,0]*distance
            wz=self.cam[2]+rays[:,:,2]*distance
            distance=np.clip((ground_height(wx,wz)-self.cam[1])/denominator,0,90)
        wx=self.cam[0]+rays[:,:,0]*distance
        wz=self.cam[2]+rays[:,:,2]*distance
        path=np.exp(-np.minimum((wx/1.8)**4,70))
        shade=30-5*path+3*np.sin(wx*.44+wz*.31)+2*np.sin(wz*.83-wx*.51)
        shade+=1.6*np.sin(wx*37+wz*43)*np.sin(wx*47-wz*20)
        shade+=2*np.sin(wx*8-wz*3)*np.sin(wx*4+wz*9)
        fog=np.exp(-distance*.025)
        color=(shade[:,:,None]*np.array([.94,.98,1])*fog[:,:,None]
               +np.array([34,40,42])*(1-fog[:,:,None]))
        floor=Image.fromarray(np.uint8(np.clip(color,0,255))).resize((W,H),Image.Resampling.BILINEAR)
        mask=Image.fromarray(np.uint8((rays[:,:,1]<-.015)*255)).resize((W,H),Image.Resampling.BILINEAR)
        self.image.paste(floor,(0,0),mask)
        # Faint atmospheric shafts are rooted in the scene.
        for x,z in [(-6,23),(5,41),(-8,60)]:
            p,depth=self.project([(x,16,z),(x-1.8,0,z-3),(x+.6,0,z-3)])
            if p:self.g.polygon(p,fill=(3,4,4))

    def draw_geometry(self, item, points, depth):
        _,color,kind,width,luminous=item
        fog=math.exp(-max(0,depth)*.025)
        c=color*fog+np.array([34,40,42])*(1-fog)
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
        pos[1]=ground_height(pos[0],pos[2])+.006*math.sin(t*1.6+pos[2])
        col=RED if state in ('red','diffusing') else BLUE
        diffusion=float(smooth(subject.get('diffuseAt',100),subject.get('goneAt',103),t))
        visibility=1-diffusion
        body=(np.array([83,91,96])+col*.44)*visibility+np.array([34,40,42])*diffusion
        gear=body*.51
        # A faceless, shaded game mannequin with a protective shell and vest.
        sway=.011*math.sin(t*1.9+pos[2])
        joints={'head':(sway,2.04,0),'neck':(sway,1.82,0),'hip':(0,.99,0),
                'ls':(-.29+sway,1.70,0),'rs':(.29+sway,1.70,0),
                'le':(-.39,1.37,.01),'re':(.38,1.37,.04),
                'lh':(-.35,1.05,-.04),'rh':(.37,1.07,-.03),
                'lk':(-.18,.55,-.02),'rk':(.20,.55,.08),
                'lf':(-.22,.06,-.11),'rf':(.23,.06,-.11)}
        P={k:pos+np.array(v)*CHAR_SCALE for k,v in joints.items()}
        center,depth=self.project([pos+[0,1.3*CHAR_SCALE,0]])
        if not center:
            return
        scale=FOCAL/depth*CHAR_SCALE
        parts=[('hip','lk',.13),('hip','rk',.13),('lk','lf',.10),('rk','rf',.10),
               ('ls','le',.12),('le','lh',.085),('rs','re',.12),('re','rh',.085)]
        if visibility>.035:
            for a,b,width in parts:
                p,_=self.project([P[a],P[b]])
                if p:
                    self.line(p,body*.69,max(2,int(scale*width*2)))
                    self.line(p,body,max(2,int(scale*width*1.22)))
                    for point in p:self.disk(point,max(2,scale*width*.62),body)
            torso=[pos+np.array(v)*CHAR_SCALE for v in [(-.27+sway,1.78,0),(.27+sway,1.78,0),(.29,1.54,0),(.20,1.01,0),(-.20,1.01,0),(-.29,1.54,0)]]
            p,_=self.project(torso)
            if p:
                self.d.polygon(p,fill=rgb(body))
                self.line(p+[p[0]],col*.85*visibility,2,.12)
            def panel(vertices,color):
                points,_=self.project([pos+np.array(v)*CHAR_SCALE for v in vertices])
                if points:self.d.polygon(points,fill=rgb(color))
            panel([(-.22,1.70,-.05),(.22,1.70,-.05),(.20,1.19,-.05),(-.20,1.19,-.05)],gear)
            for x in [-.18,.035]:
                panel([(x,1.52,-.07),(x+.145,1.52,-.07),(x+.145,1.29,-.07),(x,1.29,-.07)],body*.65)
                panel([(x,1.52,-.075),(x+.145,1.52,-.075),(x+.145,1.48,-.075),(x,1.48,-.075)],body*.90)
            panel([(-.24,1.13,-.07),(.24,1.13,-.07),(.24,1.04,-.07),(-.24,1.04,-.07)],gear*.70)
            panel([(-.04,1.14,-.08),(.04,1.14,-.08),(.04,1.04,-.08),(-.04,1.04,-.08)],body*.90)
            self.world_line([pos+[-.18*CHAR_SCALE,1.76*CHAR_SCALE,-.075],pos+[-.16*CHAR_SCALE,1.30*CHAR_SCALE,-.075]],body*.84,3)
            self.world_line([pos+[.18*CHAR_SCALE,1.76*CHAR_SCALE,-.075],pos+[.16*CHAR_SCALE,1.30*CHAR_SCALE,-.075]],body*.84,3)
            head,_=self.project([P['head']])
            if head:
                x,y=head[0];rx,ry=scale*.175,scale*.19
                self.d.ellipse((x-rx,y-ry,x+rx,y+ry),fill=rgb(body*.78),outline=rgb(col*.75*visibility),width=2)
                self.d.pieslice((x-rx*1.1,y-ry*1.16,x+rx*1.1,y+ry*.73),180,360,fill=rgb(gear*.94))
                self.d.rectangle((x-rx*1.13,y-ry*.08,x+rx*1.13,y+ry*.06),fill=rgb(body*.56))
                self.d.rounded_rectangle((x-rx*.74,y+ry*.1,x+rx*.74,y+ry*.49),radius=2,fill=rgb(body*.37))
            for k in ['lk','rk']:
                p,_=self.project([P[k]])
                if p:self.disk(p[0],max(2,scale*.078),gear)
            for k in ['lf','rf']:
                p,_=self.project([P[k]])
                if p:
                    x,y=p[0];self.d.rounded_rectangle((x-scale*.11,y-scale*.08,x+scale*.13,y+scale*.04),radius=2,fill=rgb(gear*.75))
        if subject['id']=='T1' and t>11:
            # A brief trace at the synthetic marker's feet replaces large rings.
            phase=(t*.8)%1
            amp=(1-diffusion)*float(smooth(10,16,t))*max(0,1-phase*3)
            pts=[pos+[.42*math.cos(a),.025,.28*math.sin(a)] for a in np.linspace(.12,math.pi*.9,40)]
            p,_=self.project(pts)
            if p:self.effect_line(p,col*amp*.22,1)
        if state=='diffusing':
            age=t-subject['diffuseAt']
            for i,v in enumerate(FRAGMENTS[:70]):
                origin=pos+np.array([v[0]*.17,.15+(i%21)/13,.03])
                offset=np.array([v[1]*age*.16,age*(.26+abs(v[2])*.11),v[3]*age*.10])
                p,z=self.project([origin+offset,origin+offset+[0,.028,0]])
                if p:self.effect_line(p,(RED*.35+WHITE*.25)*(1-diffusion),1)

    def particles(self):
        for i,m in enumerate(MOTES):
            p,z=self.project([m+[.035*math.sin(self.t+i),.07*math.sin(self.t*.3+i),0]])
            if p:
                flicker=.45+.25*math.sin(self.t*.9+i*3)
                self.disk(p[0],1 if z>8 else 1.5,np.array([97,133,128])*flicker*math.exp(-z*.028),.4)

    def pulses(self):
        t=self.t
        target=np.array(SCENE['subjects'][1]['position'])+[0,ground_height(.5,15.5)+1.3*CHAR_SCALE,0]
        end,depth=self.project([target])
        if not end:return
        ex,ey=end[0]
        for event in SCENE['projectiles']:
            start,arrival=event['launch'],event['arrival']
            if start<=t<=arrival:
                q=(t-start)/(arrival-start)
                port=event['port']
                p0=np.array([W/2+port*W*.045,H*1.04])
                p1=np.array([W/2+port*W*.075,H*.76])
                p2=np.array([ex-port*18,ey+44])
                p3=np.array([ex,ey])
                def bezier(s):return (1-s)**3*p0+3*(1-s)**2*s*p1+3*(1-s)*s*s*p2+s**3*p3
                tail=max(0,q-.10)
                pts=[tuple(bezier(s)) for s in np.linspace(tail,q,40)]
                self.effect_line(pts,[127,161,170],4)
                self.effect_line(pts,[227,233,217],2)
                head=bezier(q)
                self.disk(head,3,[239,242,219],.45)
            age=t-arrival
            if 0<=age<.28:
                fade=(1-age/.28)
                r=4+age*58
                for a in np.linspace(0,math.tau,9)[:-1]:
                    self.effect_line([(ex+math.cos(a)*r*.3,ey+math.sin(a)*r*.3),(ex+math.cos(a)*r,ey+math.sin(a)*r)],WHITE*fade,2)

    def reticle(self):
        t=self.t
        lock=float(smooth(12,18,t))*(1-float(smooth(23.5,27,t)))
        red=float(smooth(18.96,19.12,t))*(1-float(smooth(24,26,t)))
        col=(WHITE*(1-lock)+BLUE*lock)*(1-red)+RED*red
        col=col*.72+WHITE*.16
        radius=23-4*lock+.35*math.sin(t*1.8)
        x,y=W/2,H/2
        # A single reticle formed from open hyperbolic arcs. No other HUD.
        for angle in [0,math.pi/2,math.pi,math.pi*1.5]:
            pts=[]
            for u in np.linspace(-.76,.76,28):
                a=radius+6*(math.cosh(u*1.8)-1)
                b=8*math.sinh(u)
                pts.append((x+a*math.cos(angle)-b*math.sin(angle),y+a*math.sin(angle)+b*math.cos(angle)))
            self.d.line(pts,fill=(8,11,12),width=4,joint='curve')
            self.line(pts,col,1,.12)
        self.disk((x,y),1,col,.12)
        for dx,dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            self.line([(x+dx*7,y+dy*7),(x+dx*12,y+dy*12)],WHITE*.72,1)

    def finish(self):
        # Thermal-style halation, a soft vignette, and fine fixed-pattern noise.
        bloom=self.light.resize((W//4,H//4),Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(3.2))
        bloom=bloom.resize((W,H),Image.Resampling.BILINEAR)
        im=ImageChops.add(ImageChops.add(self.image,self.effects),bloom,scale=1)
        arr=np.asarray(im).astype(np.float32)
        arr*=vignette[:,:,None]
        # Restrained sensor scan rows, with no full-frame flashes or strobes.
        arr[::4]*=.974
        im=Image.fromarray(np.uint8(np.clip(arr,0,255)))
        return im


def draw_frame(t):
    f=Frame(t)
    f.ground()
    visible=[]
    for depth,i,p in MESH.project(f.cam,f.matrix,FOCAL,W,H):
        visible.append((depth,'geometry',GEOMETRY[i],p))
    for subject in SCENE['subjects']:
        x,_,z=subject['position']
        p,depth=f.project([np.array(subject['position'])+[0,ground_height(x,z)+1.3*CHAR_SCALE,0]])
        if p:visible.append((depth,'subject',subject,None))
    for depth,kind,item,points in sorted(visible,key=lambda v:v[0],reverse=True):
        if kind=='geometry':f.draw_geometry(item,points,depth)
        else:f.mannequin(item)
    f.particles()
    f.pulses()
    f.reticle()
    return f.finish()


def frame_bytes(index):
    return draw_frame(index/FPS).tobytes()


def synth_audio(path):
    sr=48000
    t=np.arange(sr*DURATION,dtype=np.float64)/sr
    rng=np.random.default_rng(SCENE['seed']+1)
    out=np.zeros((len(t),2),np.float64)
    # Organic pump + robotic respiration: deliberately synthetic, non-vocal.
    pad=.010*np.sin(2*np.pi*47*t+.30*np.sin(t*.41))+.006*np.sin(2*np.pi*71*t)
    air=rng.normal(0,1,len(t))
    air=np.convolve(air,np.ones(28)/28,mode='same')
    breath=(.5+.5*np.sin(t*math.tau/4.8))**3
    bed=pad+air*(.037+.060*breath)
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

    for at in np.arange(.9,42,.72):
        if 14<at<27:continue
        q=np.arange(int(.22*sr))/sr
        step=.16*np.exp(-q*25)*np.sin(math.tau*(62*q-65*q*q))
        step+=.043*rng.normal(0,1,len(q))*np.exp(-q*22)
        add(at,step,(-1 if int(at/.72)%2 else 1)*.33,'step')
    for at in np.arange(1,39,1.25):
        add(at,pulse(.14,82,46,.035),-.08,'pump')
        add(at+.19,pulse(.11,67,40,.025),.08)
    at=10.0
    while at<23:
        proximity=float(smooth(10,19,at))
        interval=.95-.66*proximity
        pos,yaw,_=camera(at)
        angle=math.atan2(.5-pos[0],15.5-pos[2])-yaw
        add(at,pulse(.065,610+proximity*190,650+proximity*210,.056),np.clip(angle*2,-.7,.7),'beacon')
        at+=interval
    add(19,pulse(.17,310,455,.070,.025),0,'red-state')
    for e in SCENE['projectiles']:
        q=np.arange(int(.26*sr))/sr
        envelope=(1-np.exp(-q*600))*np.exp(-q*25)
        signal=envelope*(.29*np.sin(math.tau*(140*q-110*q*q))+.10*rng.normal(0,1,len(q)))
        add(e['launch'],signal,e['port']*.23,e['id']+'-launch')
        add(e['launch']+.085,pulse(.07,830,460,.056,.08),e['port']*.22)
        add(e['arrival'],pulse(.19,460,170,.15,.13),0,e['id']+'-arrival')
        add(e['arrival']+.09,pulse(.18,540,340,.033),-e['port']*.25)
    add(23,pulse(1.3,330,110,.075,.14),0,'diffuse')
    for at in np.arange(23.1,25.2,.17):
        add(at,pulse(.16,1100-(at-23)*300,850,.015),math.sin(at*8)*.5)
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
    parser.add_argument('--workers',type=int,default=4,choices=range(1,5),help='Parallel frame renderers, 1–4')
    args=parser.parse_args()
    checks=validate_scene()
    work=args.work_dir or Path(tempfile.mkdtemp(prefix='muddog-fleer-'))
    work.mkdir(parents=True,exist_ok=True)
    reviews=[0,7,16.8,19.5,20.92,21.7,22.4,23.8,32,(DURATION*FPS-1)/FPS]
    sheet=Image.new('RGB',(W,H//2*5))
    for i,t in enumerate(reviews):
        im=draw_frame(t)
        im.save(work/f'frame-{t:05.2f}.png')
        tile=im.resize((W//2,H//2),Image.Resampling.LANCZOS)
        sheet.paste(tile,((i%2)*W//2,(i//2)*H//2))
    sheet.save(work/'contact-sheet.jpg',quality=93)
    print(json.dumps({'checks':checks,'preview':str(work/'contact-sheet.jpg')}),flush=True)
    if args.preview:return
    audio=synth_audio(work/'audio.wav')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{W}x{H}','-r',str(FPS),'-i','-','-i',str(work/'audio.wav'),
         '-c:v','libx264','-preset','slow','-crf','24','-pix_fmt','yuv420p',
         '-maxrate','1900k','-bufsize','3800k',
         '-c:a','aac','-b:a','192k','-t',str(DURATION),'-movflags','+faststart',
         '-map_metadata','-1',str(args.output)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        if args.workers==1:
            for i in range(DURATION*FPS):
                process.stdin.write(frame_bytes(i))
                if i%(FPS*2)==0:print(f'Rendered {i//FPS:02d} / {DURATION} seconds',flush=True)
        else:
            context=multiprocessing.get_context('fork' if sys.platform!='win32' else 'spawn')
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=context) as pool:
                for i,raw in enumerate(pool.map(frame_bytes,range(DURATION*FPS),chunksize=1)):
                    process.stdin.write(raw)
                    if i%(FPS*2)==0:print(f'Rendered {i//FPS:02d} / {DURATION} seconds',flush=True)
    finally:
        process.stdin.close()
    if process.wait():raise RuntimeError('FFmpeg failed')
    manifest={'format':'Muddog-fleer/video-v2','scene':SCENE,'checks':checks,'audio':audio,
              'video':{'file':args.output.name,'width':W,'height':H,'fps':FPS,'duration':DURATION,
                       'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()},
              'renderer':'Shaded forest mesh, uneven terrain, synthetic mannequins, batched perspective projection and drawn pulse paths.',
              'textPolicy':'No titles, captions, lettering, numbers, logos or subtitle tracks. Reticle only.'}
    args.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(str(args.output),flush=True)
    if not args.work_dir:shutil.rmtree(work)


if __name__=='__main__':
    main()
