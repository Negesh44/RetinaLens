import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { RotateCcw, ZoomIn, Eye, Sparkles, Layers, Sliders } from 'lucide-react';

export default function ThreeRetinaVisualizer({
  fundusUrl,
  heatmapUrl,
  blendAlpha = 0.55,
}) {
  const mountRef = useRef(null);
  const [elevation, setElevation] = useState(0.25);
  const [showWireframe, setShowWireframe] = useState(false);
  const [isRotating, setIsRotating] = useState(true);

  const sceneRef = useRef(null);
  const meshRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 450;
    const height = container.clientHeight || 450;

    // 1. Scene, Camera, Renderer
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 0, 3.2);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    rendererRef.current = renderer;
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    // 2. Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.4);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 2.0);
    dirLight.position.set(2, 3, 4);
    scene.add(dirLight);

    const backLight = new THREE.PointLight(0x0284c7, 3.0, 10);
    backLight.position.set(-2, -2, -2);
    scene.add(backLight);

    // 3. Load Fundus & Heatmap Textures
    const textureLoader = new THREE.TextureLoader();

    // Composite fundus + heatmap onto a dynamic canvas
    const compCanvas = document.createElement('canvas');
    compCanvas.width = 512;
    compCanvas.height = 512;
    const ctx = compCanvas.getContext('2d');

    const fundusImg = new Image();
    fundusImg.crossOrigin = "anonymous";
    fundusImg.src = fundusUrl;

    const heatImg = new Image();
    heatImg.crossOrigin = "anonymous";
    heatImg.src = heatmapUrl || fundusUrl;

    let compTexture = new THREE.CanvasTexture(compCanvas);

    const updateTexture = () => {
      ctx.clearRect(0, 0, 512, 512);
      if (fundusImg.complete && fundusImg.naturalWidth > 0) {
        ctx.globalAlpha = 1.0;
        ctx.drawImage(fundusImg, 0, 0, 512, 512);
      }
      if (heatImg.complete && heatImg.naturalWidth > 0 && heatmapUrl) {
        ctx.globalAlpha = blendAlpha;
        ctx.drawImage(heatImg, 0, 0, 512, 512);
      }
      compTexture.needsUpdate = true;
    };

    fundusImg.onload = updateTexture;
    heatImg.onload = updateTexture;

    // 4. Construct 3D Retinal Hemisphere Cup
    // In human anatomy, the retina lines the inside of a concave posterior sphere (the fundus cup)
    const hemisphereGeo = new THREE.SphereGeometry(
      1.2,
      64,
      64,
      0,
      Math.PI * 2,
      0,
      Math.PI * 0.65 // Concave hemisphere cup
    );

    // Displacement for lesion elevation
    const material = new THREE.MeshStandardMaterial({
      map: compTexture,
      side: THREE.DoubleSide,
      roughness: 0.35,
      metalness: 0.1,
      wireframe: showWireframe,
    });

    const retinaMesh = new THREE.Mesh(hemisphereGeo, material);
    retinaMesh.rotation.x = Math.PI * 0.95; // Face opening towards camera
    meshRef.current = retinaMesh;
    scene.add(retinaMesh);

    // Surrounding Scleral Rim Ring
    const rimGeo = new THREE.TorusGeometry(1.22, 0.025, 16, 64);
    const rimMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.2, metalness: 0.5 });
    const rim = new THREE.Mesh(rimGeo, rimMat);
    rim.position.z = -0.05;
    scene.add(rim);

    // 5. Mouse Orbit / Drag Interaction
    let isDragging = false;
    let prevMouseX = 0;
    let prevMouseY = 0;

    const handleMouseDown = (e) => {
      isDragging = true;
      setIsRotating(false);
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    };

    const handleMouseMove = (e) => {
      if (!isDragging) return;
      const deltaX = e.clientX - prevMouseX;
      const deltaY = e.clientY - prevMouseY;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;

      retinaMesh.rotation.y += deltaX * 0.008;
      retinaMesh.rotation.x += deltaY * 0.008;
    };

    const handleMouseUp = () => {
      isDragging = false;
    };

    const handleWheel = (e) => {
      e.preventDefault();
      camera.position.z = THREE.MathUtils.clamp(camera.position.z + e.deltaY * 0.003, 1.8, 6.0);
    };

    const domElement = renderer.domElement;
    domElement.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    domElement.addEventListener('wheel', handleWheel, { passive: false });

    // Resize handling
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // 6. Animation Loop
    let animationId;
    const animate = () => {
      animationId = requestAnimationFrame(animate);

      if (isRotating && !isDragging) {
        retinaMesh.rotation.y += 0.004;
      }

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationId);
      domElement.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      domElement.removeEventListener('wheel', handleWheel);
      window.removeEventListener('resize', handleResize);

      if (container && renderer.domElement) {
        container.innerHTML = "";
      }
      hemisphereGeo.dispose();
      material.dispose();
      compTexture.dispose();
      rimGeo.dispose();
      rimMat.dispose();
      renderer.dispose();
    };
  }, [fundusUrl, heatmapUrl, blendAlpha, showWireframe]);

  const handleResetView = () => {
    if (meshRef.current) {
      meshRef.current.rotation.set(Math.PI * 0.95, 0, 0);
    }
    if (cameraRef.current) {
      cameraRef.current.position.set(0, 0, 3.2);
    }
    setIsRotating(true);
  };

  return (
    <div className="relative w-full h-[450px] bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 shadow-2xl flex flex-col">
      {/* 3D Canvas Mount Point */}
      <div ref={mountRef} className="flex-1 w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Floating 3D Control HUD */}
      <div className="absolute top-4 left-4 z-10 bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-700/80 text-xs text-white flex items-center space-x-2 shadow-lg">
        <Sparkles className="w-3.5 h-3.5 text-sky-400" />
        <span className="font-semibold">3D Spherical Fundus & Grad-CAM Projection</span>
      </div>

      <div className="absolute top-4 right-4 z-10 flex items-center space-x-2">
        <button
          onClick={handleResetView}
          className="p-2 bg-slate-900/80 hover:bg-slate-800 backdrop-blur-md text-slate-200 hover:text-white rounded-xl border border-slate-700 transition-colors shadow-lg"
          title="Reset Camera Orientation"
        >
          <RotateCcw className="w-4 h-4" />
        </button>

        <button
          onClick={() => setShowWireframe(!showWireframe)}
          className={`px-3 py-2 backdrop-blur-md text-xs font-semibold rounded-xl border transition-all shadow-lg flex items-center space-x-1.5 ${
            showWireframe
              ? "bg-sky-600 text-white border-sky-500"
              : "bg-slate-900/80 text-slate-200 hover:text-white border-slate-700"
          }`}
          title="Toggle 3D Anatomical Wireframe"
        >
          <Layers className="w-3.5 h-3.5" />
          <span>{showWireframe ? "Solid Map" : "Wireframe"}</span>
        </button>
      </div>

      {/* Interaction Hint Bottom Overlay */}
      <div className="absolute bottom-3 inset-x-4 z-10 bg-slate-900/70 backdrop-blur-md px-4 py-2 rounded-xl border border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400">
        <span className="flex items-center space-x-1.5">
          <Eye className="w-3.5 h-3.5 text-sky-400" />
          <span>Click &amp; drag to rotate posterior retinal cup &bull; Scroll wheel to zoom depth</span>
        </span>
        <button
          onClick={() => setIsRotating(!isRotating)}
          className="text-sky-400 hover:text-sky-300 font-bold"
        >
          {isRotating ? "Pause Auto-Spin" : "Resume Auto-Spin"}
        </button>
      </div>
    </div>
  );
}
