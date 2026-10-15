import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';

const EYE_R = 1.2;
const IRIS_R = 0.55;
const LIMBUS_ANGLE = Math.asin(IRIS_R / EYE_R); // angular radius of the corneal opening
const LIMBUS_Z = EYE_R * Math.cos(LIMBUS_ANGLE);
const HUD_COLOR = 0xa78bfa;

const rand = (a, b) => a + Math.random() * (b - a);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function makeCanvas(w, h = w) {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  return [canvas, canvas.getContext('2d')];
}

function toTexture(canvas) {
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

function makeIrisTexture() {
  const S = 1024;
  const c = S / 2;
  const R = c;
  const [canvas, ctx] = makeCanvas(S);

  const base = ctx.createRadialGradient(c, c, 0, c, c, R);
  base.addColorStop(0, '#1a1006');
  base.addColorStop(0.22, '#a8742a');
  base.addColorStop(0.36, '#cf9f4a');
  base.addColorStop(0.47, '#3d93ad');
  base.addColorStop(0.72, '#1c628a');
  base.addColorStop(0.9, '#0c3354');
  base.addColorStop(1, '#030b16');
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, S, S);

  // Stromal fibers: slightly wandering radial strands, light and dark.
  for (let i = 0; i < 1100; i++) {
    let ang = rand(0, Math.PI * 2);
    const r0 = R * rand(0.2, 0.36);
    const r1 = R * rand(0.6, 0.95);
    const light = Math.random() < 0.6;
    ctx.strokeStyle = light
      ? `rgba(175, 228, 245, ${rand(0.06, 0.26)})`
      : `rgba(4, 20, 36, ${rand(0.1, 0.32)})`;
    ctx.lineWidth = rand(0.6, 2.2);
    ctx.beginPath();
    for (let s = 0; s <= 8; s++) {
      const r = r0 + ((r1 - r0) * s) / 8;
      ang += rand(-0.018, 0.018);
      const x = c + Math.cos(ang) * r;
      const y = c + Math.sin(ang) * r;
      if (s === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Crypts: small dark lacunae in the mid-periphery.
  for (let i = 0; i < 80; i++) {
    const a = rand(0, Math.PI * 2);
    const r = R * rand(0.45, 0.82);
    ctx.save();
    ctx.translate(c + Math.cos(a) * r, c + Math.sin(a) * r);
    ctx.rotate(a);
    ctx.fillStyle = `rgba(3, 14, 28, ${rand(0.25, 0.55)})`;
    ctx.beginPath();
    ctx.ellipse(0, 0, R * rand(0.02, 0.05), R * rand(0.007, 0.016), 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // Collarette: the wavy golden ring around the pupil zone.
  ctx.strokeStyle = 'rgba(244, 204, 124, 0.6)';
  ctx.lineWidth = R * 0.016;
  ctx.beginPath();
  for (let i = 0; i <= 240; i++) {
    const a = (i / 240) * Math.PI * 2;
    const r = R * (0.41 + Math.sin(a * 15) * 0.018 + Math.sin(a * 4) * 0.014);
    const x = c + Math.cos(a) * r;
    const y = c + Math.sin(a) * r;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.stroke();

  // Dark limbal ring at the iris edge.
  const limbal = ctx.createRadialGradient(c, c, R * 0.8, c, c, R);
  limbal.addColorStop(0, 'rgba(2, 8, 16, 0)');
  limbal.addColorStop(1, 'rgba(2, 8, 16, 0.96)');
  ctx.fillStyle = limbal;
  ctx.fillRect(0, 0, S, S);

  return toTexture(canvas);
}

// Sclera UVs run from the limbus (canvas top) to the back of the globe (canvas bottom).
function makeScleraTexture() {
  const W = 1024;
  const H = 512;
  const [canvas, ctx] = makeCanvas(W, H);

  const base = ctx.createLinearGradient(0, 0, 0, H);
  base.addColorStop(0, '#d5dde7');
  base.addColorStop(0.06, '#f7f5f2');
  base.addColorStop(0.45, '#f0e4dd');
  base.addColorStop(1, '#d6b0a6');
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, W, H);

  const drawVein = (x, y, yEnd, width, depth) => {
    let w = width;
    while (y > yEnd && w > 0.25) {
      const nx = x + rand(-9, 9);
      const ny = y - rand(6, 14);
      const fade = clamp((y - yEnd) / (H * 0.25), 0, 1);
      ctx.strokeStyle = `rgba(${Math.round(rand(160, 200))}, 38, 48, ${0.5 * fade})`;
      ctx.lineWidth = w;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(nx, ny);
      ctx.stroke();
      if (depth < 2 && Math.random() < 0.07) {
        drawVein(nx, ny, yEnd + rand(0, H * 0.12), w * 0.6, depth + 1);
      }
      x = nx;
      y = ny;
      w *= 0.975;
    }
  };
  for (let i = 0; i < 48; i++) {
    drawVein(rand(24, W - 24), H * rand(0.38, 0.62), H * rand(0.05, 0.16), rand(1.2, 2.8), 0);
  }

  return toTexture(canvas);
}

function makeGlowTexture(inner, outer) {
  const S = 256;
  const [canvas, ctx] = makeCanvas(S);
  const g = ctx.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
  g.addColorStop(0, inner);
  g.addColorStop(1, outer);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, S, S);
  return toTexture(canvas);
}

function makeBandTexture() {
  const [canvas, ctx] = makeCanvas(4, 128);
  const g = ctx.createLinearGradient(0, 0, 0, 128);
  g.addColorStop(0, 'rgba(167, 139, 250, 0)');
  g.addColorStop(0.5, 'rgba(196, 181, 253, 1)');
  g.addColorStop(1, 'rgba(167, 139, 250, 0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 4, 128);
  return toTexture(canvas);
}

// Radar trail: bright at the leading edge, fading out over the trailing quarter turn.
function makeSweepTexture() {
  const S = 512;
  const [canvas, ctx] = makeCanvas(S);
  const g = ctx.createConicGradient(0, S / 2, S / 2);
  g.addColorStop(0, 'rgba(196, 181, 253, 0.9)');
  g.addColorStop(0.004, 'rgba(167, 139, 250, 0.45)');
  g.addColorStop(0.25, 'rgba(167, 139, 250, 0)');
  g.addColorStop(1, 'rgba(167, 139, 250, 0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, S, S);
  return toTexture(canvas);
}

const additive = (opts) =>
  new THREE.MeshBasicMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    ...opts,
  });

export default function ThreeEyeScanner({ isScanning = false, size = 260 }) {
  const mountRef = useRef(null);
  const scanningRef = useRef(isScanning);

  useEffect(() => {
    scanningRef.current = isScanning;
  }, [isScanning]);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return undefined;

    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

    // Renderer, scene, camera
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(size, size);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
    camera.position.z = 5.6;

    const pmrem = new THREE.PMREMGenerator(renderer);
    const envTexture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = envTexture;

    scene.add(new THREE.AmbientLight(0xffffff, 0.35));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(-2, 3, 4);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xa78bfa, 2.2);
    rim.position.set(3, 1, -3);
    scene.add(rim);

    // Halo behind the eye
    const haloMat = new THREE.SpriteMaterial({
      map: makeGlowTexture('rgba(167, 139, 250, 0.55)', 'rgba(167, 139, 250, 0)'),
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      transparent: true,
    });
    const halo = new THREE.Sprite(haloMat);
    halo.scale.set(4.4, 4.4, 1);
    halo.position.z = -1.5;
    scene.add(halo);

    // Eyeball: sclera with an opening for the cornea, iris, pupil, and cornea dome
    const eye = new THREE.Group();
    scene.add(eye);

    const scleraGeo = new THREE.SphereGeometry(EYE_R, 128, 64, 0, Math.PI * 2, LIMBUS_ANGLE, Math.PI - LIMBUS_ANGLE);
    scleraGeo.rotateX(Math.PI / 2);
    eye.add(
      new THREE.Mesh(
        scleraGeo,
        new THREE.MeshPhysicalMaterial({
          map: makeScleraTexture(),
          roughness: 0.32,
          clearcoat: 0.8,
          clearcoatRoughness: 0.12,
          sheen: 0.3,
          sheenColor: new THREE.Color(0xffe4e6),
        }),
      ),
    );

    const iris = new THREE.Mesh(
      new THREE.CircleGeometry(IRIS_R * 1.01, 96),
      new THREE.MeshStandardMaterial({ map: makeIrisTexture(), roughness: 0.55 }),
    );
    iris.position.z = LIMBUS_Z - 0.008;
    eye.add(iris);

    const pupil = new THREE.Mesh(
      new THREE.CircleGeometry(IRIS_R * 0.34, 64),
      new THREE.MeshBasicMaterial({
        map: makeGlowTexture('rgba(1, 3, 8, 1)', 'rgba(1, 3, 8, 0)'),
        transparent: true,
        depthWrite: false,
      }),
    );
    pupil.position.z = LIMBUS_Z - 0.004;
    eye.add(pupil);

    const corneaR = 0.8;
    const corneaAngle = Math.asin((IRIS_R * 1.03) / corneaR);
    const corneaGeo = new THREE.SphereGeometry(corneaR, 96, 32, 0, Math.PI * 2, 0, corneaAngle);
    corneaGeo.rotateX(Math.PI / 2);
    const cornea = new THREE.Mesh(
      corneaGeo,
      new THREE.MeshPhysicalMaterial({
        transmission: 1,
        thickness: 0.15,
        roughness: 0.02,
        ior: 1.376,
        clearcoat: 1,
        clearcoatRoughness: 0.02,
        envMapIntensity: 1.8,
        transparent: true,
      }),
    );
    cornea.position.z = LIMBUS_Z - corneaR * Math.cos(corneaAngle);
    eye.add(cornea);

    // HUD: segmented arcs, tick dial, inner ring with orbiting marker
    const hud = new THREE.Group();
    scene.add(hud);

    const arcs = new THREE.Group();
    const arcMat = additive({ color: HUD_COLOR, opacity: 0.75 });
    for (let k = 0; k < 3; k++) {
      arcs.add(new THREE.Mesh(new THREE.RingGeometry(1.6, 1.625, 64, 1, (k * Math.PI * 2) / 3, 1.45), arcMat));
    }
    hud.add(arcs);

    const tickPositions = [];
    for (let i = 0; i < 120; i++) {
      const a = (i / 120) * Math.PI * 2;
      const r2 = i % 10 === 0 ? 1.8 : 1.73;
      tickPositions.push(Math.cos(a) * 1.69, Math.sin(a) * 1.69, 0, Math.cos(a) * r2, Math.sin(a) * r2, 0);
    }
    const tickGeo = new THREE.BufferGeometry();
    tickGeo.setAttribute('position', new THREE.Float32BufferAttribute(tickPositions, 3));
    const tickMat = new THREE.LineBasicMaterial({ color: 0xc4b5fd, transparent: true, opacity: 0.3, depthWrite: false });
    const ticks = new THREE.LineSegments(tickGeo, tickMat);
    hud.add(ticks);

    const innerRingMat = additive({ color: HUD_COLOR, opacity: 0.25 });
    hud.add(new THREE.Mesh(new THREE.RingGeometry(1.47, 1.476, 128), innerRingMat));

    const orbiter = new THREE.Group();
    const dotMat = additive({ color: 0xede9fe, opacity: 1 });
    const dot = new THREE.Mesh(new THREE.CircleGeometry(0.035, 24), dotMat);
    const dotGlowMat = new THREE.SpriteMaterial({
      map: makeGlowTexture('rgba(196, 181, 253, 0.9)', 'rgba(167, 139, 250, 0)'),
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      transparent: true,
    });
    const dotGlow = new THREE.Sprite(dotGlowMat);
    dotGlow.scale.set(0.32, 0.32, 1);
    dot.position.x = 1.473;
    dotGlow.position.x = 1.473;
    orbiter.add(dot, dotGlow);
    hud.add(orbiter);

    // Scanning effects: radar sweep and a sweeping laser band
    const sweepMat = additive({ map: makeSweepTexture(), opacity: 0 });
    const sweep = new THREE.Mesh(new THREE.CircleGeometry(1.47, 96), sweepMat);
    sweep.position.z = 1.4;
    scene.add(sweep);

    const laserMat = additive({ color: 0xede9fe, opacity: 0 });
    const laser = new THREE.Mesh(new THREE.PlaneGeometry(3.0, 0.02), laserMat);
    const bandMat = additive({ map: makeBandTexture(), opacity: 0 });
    const band = new THREE.Mesh(new THREE.PlaneGeometry(3.0, 0.3), bandMat);
    const laserGroup = new THREE.Group();
    laserGroup.add(band, laser);
    laserGroup.position.z = 1.45;
    scene.add(laserGroup);

    // Gaze: follow the cursor within natural limits; wander with saccades when idle.
    const gaze = { x: 0, y: 0 };
    const target = { x: 0, y: 0 };
    let lastPointer = -Infinity;
    let nextSaccade = 0;

    const handlePointerMove = (e) => {
      const rect = container.getBoundingClientRect();
      const dx = (e.clientX - (rect.left + rect.width / 2)) / (window.innerWidth / 2);
      const dy = (e.clientY - (rect.top + rect.height / 2)) / (window.innerHeight / 2);
      target.x = clamp(dx, -1, 1) * 0.5;
      target.y = clamp(dy, -1, 1) * 0.38;
      lastPointer = performance.now();
    };
    window.addEventListener('pointermove', handlePointerMove, { passive: true });

    // Only render while on screen and the tab is visible.
    let onScreen = true;
    const observer = new IntersectionObserver(([entry]) => {
      onScreen = entry.isIntersecting;
    });
    observer.observe(container);

    const start = performance.now();
    let last = start;
    let scanMix = 0;
    let frameId;

    const animate = () => {
      frameId = requestAnimationFrame(animate);
      if (!onScreen || document.hidden) return;

      const now = performance.now();
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      const t = (now - start) / 1000;
      const scanning = scanningRef.current;
      scanMix += ((scanning ? 1 : 0) - scanMix) * Math.min(1, dt * 4);

      if (scanning) {
        target.x = Math.sin(t * 7) * 0.015;
        target.y = Math.cos(t * 5) * 0.015;
      } else if (!reduceMotion && now - lastPointer > 2500 && now > nextSaccade) {
        target.x = rand(-0.3, 0.3);
        target.y = rand(-0.2, 0.2);
        nextSaccade = now + rand(1100, 2800);
      }
      const follow = now - lastPointer > 2500 ? 0.22 : 0.09;
      gaze.x += (target.x - gaze.x) * follow;
      gaze.y += (target.y - gaze.y) * follow;
      eye.rotation.y = gaze.x;
      eye.rotation.x = gaze.y;
      eye.position.y = reduceMotion ? 0 : Math.sin(t * 1.2) * 0.03;

      // Pupil: gentle hippus at rest, constricts under the scanning light.
      const pupilScale = (1 + Math.sin(t * 0.9) * 0.06) * (1 - 0.35 * scanMix);
      pupil.scale.set(pupilScale, pupilScale, 1);

      const speed = reduceMotion ? 0.3 : 1;
      arcs.rotation.z += dt * (0.25 + 1.6 * scanMix) * speed;
      ticks.rotation.z -= dt * (0.06 + 0.3 * scanMix) * speed;
      orbiter.rotation.z -= dt * (0.9 + 2.2 * scanMix) * speed;
      arcMat.opacity = 0.55 + 0.35 * scanMix;
      innerRingMat.opacity = 0.2 + 0.25 * scanMix;
      haloMat.opacity = 0.75 + Math.sin(t * 1.6) * 0.12 + 0.35 * scanMix;

      sweep.rotation.z -= dt * 3.2;
      sweepMat.opacity = 0.55 * scanMix;
      laserGroup.position.y = Math.sin(t * 2.4) * 1.15;
      laserMat.opacity = 0.95 * scanMix;
      bandMat.opacity = 0.55 * scanMix;

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(frameId);
      observer.disconnect();
      window.removeEventListener('pointermove', handlePointerMove);
      scene.traverse((obj) => {
        obj.geometry?.dispose();
        const mats = Array.isArray(obj.material) ? obj.material : obj.material ? [obj.material] : [];
        mats.forEach((m) => {
          m.map?.dispose();
          m.dispose();
        });
      });
      envTexture.dispose();
      pmrem.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) container.removeChild(renderer.domElement);
    };
  }, [size]);

  return (
    <div
      ref={mountRef}
      className="relative flex select-none items-center justify-center"
      style={{ width: size, height: size }}
      aria-hidden="true"
    />
  );
}
