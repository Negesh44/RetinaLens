import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

export default function ThreeNeuralBackground() {
  const mountRef = useRef(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    // 1. Scene, Camera, Renderer
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      1000
    );
    camera.position.z = 250;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: "high-performance" });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // 2. Generate 3D Particle Constellation (Ganglion cell / neural fiber network)
    const particleCount = 180;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const velocities = [];
    const colors = new Float32Array(particleCount * 3);

    const baseColor = new THREE.Color("#7c3aed"); // Violet
    const highlightColor = new THREE.Color("#a78bfa"); // Lavender
    const accentColor = new THREE.Color("#b45309"); // Sandalwood

    for (let i = 0; i < particleCount; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 550;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 350;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 200;

      velocities.push({
        x: (Math.random() - 0.5) * 0.25,
        y: (Math.random() - 0.5) * 0.25,
        z: (Math.random() - 0.5) * 0.15,
      });

      const chosenColor = Math.random() > 0.6 ? highlightColor : (Math.random() > 0.3 ? baseColor : accentColor);
      colors[i * 3] = chosenColor.r;
      colors[i * 3 + 1] = chosenColor.g;
      colors[i * 3 + 2] = chosenColor.b;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Particle Material
    const pMaterial = new THREE.PointsMaterial({
      size: 3.5,
      vertexColors: true,
      transparent: true,
      opacity: 0.55,
      blending: THREE.AdditiveBlending,
    });

    const particles = new THREE.Points(geometry, pMaterial);
    scene.add(particles);

    // Dynamic Connections Lines Material
    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0x7c3aed,
      transparent: true,
      opacity: 0.12,
      blending: THREE.AdditiveBlending,
    });

    const maxLineSegments = (particleCount * (particleCount - 1)) / 2;
    const linePositions = new Float32Array(maxLineSegments * 6);
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(lines);

    // 3. Mouse Interaction
    let mouseX = 0;
    let mouseY = 0;
    let targetX = 0;
    let targetY = 0;

    const handleMouseMove = (e) => {
      mouseX = (e.clientX - window.innerWidth / 2) * 0.05;
      mouseY = (e.clientY - window.innerHeight / 2) * 0.05;
    };

    window.addEventListener('mousemove', handleMouseMove, { passive: true });

    // Resize Handler
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);

    // 4. Animation Loop
    let animationFrameId;
    const pos = geometry.attributes.position.array;

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      // Camera smooth damping towards mouse
      targetX += (mouseX - targetX) * 0.03;
      targetY += (mouseY - targetY) * 0.03;
      camera.position.x = targetX;
      camera.position.y = -targetY;
      camera.lookAt(scene.position);

      // Update particle positions
      let lineVertexIndex = 0;
      for (let i = 0; i < particleCount; i++) {
        pos[i * 3] += velocities[i].x;
        pos[i * 3 + 1] += velocities[i].y;
        pos[i * 3 + 2] += velocities[i].z;

        // Boundary bouncing
        if (pos[i * 3] < -280 || pos[i * 3] > 280) velocities[i].x *= -1;
        if (pos[i * 3 + 1] < -180 || pos[i * 3 + 1] > 180) velocities[i].y *= -1;
        if (pos[i * 3 + 2] < -100 || pos[i * 3 + 2] > 100) velocities[i].z *= -1;

        // Connect nearby particles with synaptic lines
        for (let j = i + 1; j < particleCount; j++) {
          const dx = pos[i * 3] - pos[j * 3];
          const dy = pos[i * 3 + 1] - pos[j * 3 + 1];
          const dz = pos[i * 3 + 2] - pos[j * 3 + 2];
          const distSq = dx * dx + dy * dy + dz * dz;

          if (distSq < 3200) { // ~56 units
            const lpos = lineGeometry.attributes.position.array;
            lpos[lineVertexIndex++] = pos[i * 3];
            lpos[lineVertexIndex++] = pos[i * 3 + 1];
            lpos[lineVertexIndex++] = pos[i * 3 + 2];
            lpos[lineVertexIndex++] = pos[j * 3];
            lpos[lineVertexIndex++] = pos[j * 3 + 1];
            lpos[lineVertexIndex++] = pos[j * 3 + 2];
          }
        }
      }

      geometry.attributes.position.needsUpdate = true;
      lineGeometry.setDrawRange(0, lineVertexIndex / 3);
      lineGeometry.attributes.position.needsUpdate = true;

      // Subtle scene rotation
      scene.rotation.y += 0.0006;
      scene.rotation.x += 0.0003;

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
      if (container && renderer.domElement) {
        container.removeChild(renderer.domElement);
      }
      geometry.dispose();
      lineGeometry.dispose();
      pMaterial.dispose();
      lineMaterial.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    <div
      ref={mountRef}
      className="fixed inset-0 pointer-events-none z-0 opacity-20 overflow-hidden"
      aria-hidden="true"
    />
  );
}
