import React, { useEffect, useRef, useState } from 'react';
import '@/App.css';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Save, Trash2, Play, Pause } from 'lucide-react';

// Utility functions for linear interpolation
const lerp = (A, B, t) => A + (B - A) * t;

const getIntersection = (A, B, C, D) => {
  const tTop = (D.x - C.x) * (A.y - C.y) - (D.y - C.y) * (A.x - C.x);
  const uTop = (C.y - A.y) * (A.x - B.x) - (C.x - A.x) * (A.y - B.y);
  const bottom = (D.y - C.y) * (B.x - A.x) - (D.x - C.x) * (B.y - A.y);

  if (bottom !== 0) {
    const t = tTop / bottom;
    const u = uTop / bottom;
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
      return {
        x: lerp(A.x, B.x, t),
        y: lerp(A.y, B.y, t),
        offset: t
      };
    }
  }
  return null;
};

const polysIntersect = (poly1, poly2) => {
  for (let i = 0; i < poly1.length; i++) {
    for (let j = 0; j < poly2.length; j++) {
      const touch = getIntersection(
        poly1[i],
        poly1[(i + 1) % poly1.length],
        poly2[j],
        poly2[(j + 1) % poly2.length]
      );
      if (touch) return true;
    }
  }
  return false;
};

// Traffic Light class
class TrafficLight {
  constructor(y, x, startingPhase = 0) {
    this.y = y;
    this.x = x;
    this.state = 'green'; // Start with green to allow initial movement
    this.timer = startingPhase;
    this.redDuration = 120; // Reduced from 200
    this.yellowDuration = 30; // Reduced from 50
    this.greenDuration = 180; // Reduced from 200
    
    // Set initial state based on starting phase
    if (startingPhase < this.greenDuration) {
      this.state = 'green';
    } else if (startingPhase < this.greenDuration + this.yellowDuration) {
      this.state = 'yellow';
    } else {
      this.state = 'red';
    }
  }

  update() {
    this.timer++;
    
    if (this.state === 'green' && this.timer > this.greenDuration) {
      this.state = 'yellow';
      this.timer = 0;
    } else if (this.state === 'yellow' && this.timer > this.yellowDuration) {
      this.state = 'red';
      this.timer = 0;
    } else if (this.state === 'red' && this.timer > this.redDuration) {
      this.state = 'green';
      this.timer = 0;
    }
  }

  draw(ctx) {
    const size = 15;
    const spacing = 5;
    
    // Traffic light box
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(this.x - size - 5, this.y - size * 3 - spacing * 2 - 5, size * 2 + 10, size * 3 + spacing * 2 + 10);
    
    // Red light
    ctx.beginPath();
    ctx.arc(this.x, this.y - size * 2 - spacing * 2, size, 0, Math.PI * 2);
    ctx.fillStyle = this.state === 'red' ? '#ef4444' : '#7f1d1d';
    ctx.fill();
    if (this.state === 'red') {
      ctx.shadowColor = '#ef4444';
      ctx.shadowBlur = 20;
      ctx.fill();
      ctx.shadowBlur = 0;
    }
    
    // Yellow light
    ctx.beginPath();
    ctx.arc(this.x, this.y - size - spacing, size, 0, Math.PI * 2);
    ctx.fillStyle = this.state === 'yellow' ? '#fbbf24' : '#78350f';
    ctx.fill();
    if (this.state === 'yellow') {
      ctx.shadowColor = '#fbbf24';
      ctx.shadowBlur = 20;
      ctx.fill();
      ctx.shadowBlur = 0;
    }
    
    // Green light
    ctx.beginPath();
    ctx.arc(this.x, this.y, size, 0, Math.PI * 2);
    ctx.fillStyle = this.state === 'green' ? '#22c55e' : '#14532d';
    ctx.fill();
    if (this.state === 'green') {
      ctx.shadowColor = '#22c55e';
      ctx.shadowBlur = 20;
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  getStopLine() {
    return this.y + 40;
  }

  shouldStop(carY, carHeight) {
    const stopLine = this.getStopLine();
    const carFront = carY - carHeight / 2;
    const detectionZone = 100;
    
    return (
      this.state === 'red' &&
      carFront < stopLine &&
      carFront > stopLine - detectionZone
    );
  }

  getStateValue() {
    // Return value for neural network: red=0, yellow=0.5, green=1
    if (this.state === 'red') return 0;
    if (this.state === 'yellow') return 0.5;
    return 1;
  }
}

// Neural Network implementation
class NeuralNetwork {
  constructor(neuronCounts) {
    this.levels = [];
    for (let i = 0; i < neuronCounts.length - 1; i++) {
      this.levels.push(new Level(neuronCounts[i], neuronCounts[i + 1]));
    }
  }

  static feedForward(givenInputs, network) {
    let outputs = Level.feedForward(givenInputs, network.levels[0]);
    for (let i = 1; i < network.levels.length; i++) {
      outputs = Level.feedForward(outputs, network.levels[i]);
    }
    return outputs;
  }

  static mutate(network, amount = 1) {
    network.levels.forEach(level => {
      for (let i = 0; i < level.biases.length; i++) {
        level.biases[i] = lerp(level.biases[i], Math.random() * 2 - 1, amount);
      }
      for (let i = 0; i < level.weights.length; i++) {
        for (let j = 0; j < level.weights[i].length; j++) {
          level.weights[i][j] = lerp(level.weights[i][j], Math.random() * 2 - 1, amount);
        }
      }
    });
  }
}

class Level {
  constructor(inputCount, outputCount) {
    this.inputs = new Array(inputCount);
    this.outputs = new Array(outputCount);
    this.biases = new Array(outputCount);
    this.weights = [];

    for (let i = 0; i < inputCount; i++) {
      this.weights[i] = new Array(outputCount);
    }

    Level.randomize(this);
  }

  static randomize(level) {
    for (let i = 0; i < level.inputs.length; i++) {
      for (let j = 0; j < level.outputs.length; j++) {
        level.weights[i][j] = Math.random() * 2 - 1;
      }
    }
    for (let i = 0; i < level.biases.length; i++) {
      level.biases[i] = Math.random() * 2 - 1;
    }
  }

  static feedForward(givenInputs, level) {
    for (let i = 0; i < level.inputs.length; i++) {
      level.inputs[i] = givenInputs[i];
    }

    for (let i = 0; i < level.outputs.length; i++) {
      let sum = 0;
      for (let j = 0; j < level.inputs.length; j++) {
        sum += level.inputs[j] * level.weights[j][i];
      }

      if (sum > level.biases[i]) {
        level.outputs[i] = 1;
      } else {
        level.outputs[i] = 0;
      }
    }
    return level.outputs;
  }
}

// Sensor class for ray-casting
class Sensor {
  constructor(car) {
    this.car = car;
    this.rayCount = 5;
    this.rayLength = 150;
    this.raySpread = Math.PI / 2;
    this.rays = [];
    this.readings = [];
    this.trafficLightReading = 1; // Default to green (safe)
  }

  update(roadBorders, traffic, trafficLights) {
    this.castRays();
    this.readings = [];
    for (let i = 0; i < this.rays.length; i++) {
      this.readings.push(this.getReading(this.rays[i], roadBorders, traffic));
    }
    
    // Check for upcoming traffic light
    this.trafficLightReading = this.getTrafficLightReading(trafficLights);
  }

  getTrafficLightReading(trafficLights) {
    // Find the nearest traffic light ahead
    let nearestLight = null;
    let minDistance = Infinity;
    
    for (let light of trafficLights) {
      const distance = light.y - this.car.y;
      // Increased detection range and added distance normalization
      if (distance > -50 && distance < 300 && distance < minDistance) {
        minDistance = distance;
        nearestLight = light;
      }
    }
    
    if (nearestLight) {
      const distance = nearestLight.y - this.car.y;
      // Return normalized value: closer to light = higher urgency
      const urgency = 1 - Math.max(0, Math.min(distance / 300, 1));
      return nearestLight.getStateValue() + (urgency * 0.1); // Add distance factor
    }
    return 1; // No light ahead, assume green
  }

  getReading(ray, roadBorders, traffic) {
    let touches = [];

    for (let i = 0; i < roadBorders.length; i++) {
      const touch = getIntersection(
        ray[0],
        ray[1],
        roadBorders[i][0],
        roadBorders[i][1]
      );
      if (touch) {
        touches.push(touch);
      }
    }

    for (let i = 0; i < traffic.length; i++) {
      const poly = traffic[i].polygon;
      for (let j = 0; j < poly.length; j++) {
        const touch = getIntersection(
          ray[0],
          ray[1],
          poly[j],
          poly[(j + 1) % poly.length]
        );
        if (touch) {
          touches.push(touch);
        }
      }
    }

    if (touches.length === 0) {
      return null;
    } else {
      const offsets = touches.map(e => e.offset);
      const minOffset = Math.min(...offsets);
      return touches.find(e => e.offset === minOffset);
    }
  }

  castRays() {
    this.rays = [];
    for (let i = 0; i < this.rayCount; i++) {
      const rayAngle =
        lerp(
          this.raySpread / 2,
          -this.raySpread / 2,
          this.rayCount === 1 ? 0.5 : i / (this.rayCount - 1)
        ) + this.car.angle;

      const start = { x: this.car.x, y: this.car.y };
      const end = {
        x: this.car.x - Math.sin(rayAngle) * this.rayLength,
        y: this.car.y - Math.cos(rayAngle) * this.rayLength
      };
      this.rays.push([start, end]);
    }
  }

  draw(ctx) {
    for (let i = 0; i < this.rayCount; i++) {
      let end = this.rays[i][1];
      if (this.readings[i]) {
        end = this.readings[i];
      }

      ctx.beginPath();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#4ade80';
      ctx.moveTo(this.rays[i][0].x, this.rays[i][0].y);
      ctx.lineTo(end.x, end.y);
      ctx.stroke();

      ctx.beginPath();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ef4444';
      ctx.moveTo(this.rays[i][1].x, this.rays[i][1].y);
      ctx.lineTo(end.x, end.y);
      ctx.stroke();
    }
  }
}

// Controls class
class Controls {
  constructor(type) {
    this.forward = false;
    this.left = false;
    this.right = false;
    this.reverse = false;

    switch (type) {
      case 'KEYS':
        this.addKeyboardListeners();
        break;
      case 'DUMMY':
        this.forward = true;
        break;
    }
  }

  addKeyboardListeners() {
    document.onkeydown = (event) => {
      switch (event.key) {
        case 'ArrowLeft':
          this.left = true;
          break;
        case 'ArrowRight':
          this.right = true;
          break;
        case 'ArrowUp':
          this.forward = true;
          break;
        case 'ArrowDown':
          this.reverse = true;
          break;
      }
    };
    document.onkeyup = (event) => {
      switch (event.key) {
        case 'ArrowLeft':
          this.left = false;
          break;
        case 'ArrowRight':
          this.right = false;
          break;
        case 'ArrowUp':
          this.forward = false;
          break;
        case 'ArrowDown':
          this.reverse = false;
          break;
      }
    };
  }
}

// Car class
class Car {
  constructor(x, y, width, height, controlType, maxSpeed = 3) {
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;

    this.speed = 0;
    this.acceleration = 0.25; // Increased for better movement
    this.maxSpeed = maxSpeed;
    this.friction = 0.06; // Slightly increased friction
    this.angle = 0;
    this.damaged = false;
    this.stuckTimer = 0; // Track if car is stuck
    this.lastY = y;

    this.useBrain = controlType === 'AI';

    if (controlType !== 'DUMMY') {
      this.sensor = new Sensor(this);
      // Updated network: 5 sensors + 1 traffic light state = 6 inputs
      // Increased hidden layer for better learning
      this.brain = new NeuralNetwork([6, 10, 4]);
    }
    this.controls = new Controls(controlType);

    this.polygon = [];
    this.createPolygon();
  }

  update(roadBorders, traffic, trafficLights) {
    if (!this.damaged) {
      this.move();
      this.createPolygon();
      this.damaged = this.assessDamage(roadBorders, traffic);
      
      // Anti-stuck mechanism: check if car hasn't moved
      if (Math.abs(this.y - this.lastY) < 0.1) {
        this.stuckTimer++;
        // If stuck for too long, give a small nudge
        if (this.stuckTimer > 100 && this.useBrain) {
          this.speed = 0.5; // Small forward push
          this.stuckTimer = 0;
        }
      } else {
        this.stuckTimer = 0;
      }
      this.lastY = this.y;
    }
    if (this.sensor) {
      this.sensor.update(roadBorders, traffic, trafficLights);
      const offsets = this.sensor.readings.map(s =>
        s == null ? 0 : 1 - s.offset
      );
      // Add traffic light state to inputs
      const inputs = [...offsets, this.sensor.trafficLightReading];
      const outputs = NeuralNetwork.feedForward(inputs, this.brain);

      if (this.useBrain) {
        this.controls.forward = outputs[0];
        this.controls.left = outputs[1];
        this.controls.right = outputs[2];
        this.controls.reverse = outputs[3];
      }
    }
  }

  assessDamage(roadBorders, traffic) {
    for (let i = 0; i < roadBorders.length; i++) {
      if (polysIntersect(this.polygon, [roadBorders[i][0], roadBorders[i][1]].map(p => p))) {
        return true;
      }
    }
    for (let i = 0; i < traffic.length; i++) {
      if (polysIntersect(this.polygon, traffic[i].polygon)) {
        return true;
      }
    }
    return false;
  }

  createPolygon() {
    const points = [];
    const rad = Math.hypot(this.width, this.height) / 2;
    const alpha = Math.atan2(this.width, this.height);
    points.push({
      x: this.x - Math.sin(this.angle - alpha) * rad,
      y: this.y - Math.cos(this.angle - alpha) * rad
    });
    points.push({
      x: this.x - Math.sin(this.angle + alpha) * rad,
      y: this.y - Math.cos(this.angle + alpha) * rad
    });
    points.push({
      x: this.x - Math.sin(Math.PI + this.angle - alpha) * rad,
      y: this.y - Math.cos(Math.PI + this.angle - alpha) * rad
    });
    points.push({
      x: this.x - Math.sin(Math.PI + this.angle + alpha) * rad,
      y: this.y - Math.cos(Math.PI + this.angle + alpha) * rad
    });
    this.polygon = points;
  }

  move() {
    if (this.controls.forward) {
      this.speed += this.acceleration;
    }
    if (this.controls.reverse) {
      this.speed -= this.acceleration;
    }

    if (this.speed > this.maxSpeed) {
      this.speed = this.maxSpeed;
    }
    if (this.speed < -this.maxSpeed / 2) {
      this.speed = -this.maxSpeed / 2;
    }

    if (this.speed > 0) {
      this.speed -= this.friction;
    }
    if (this.speed < 0) {
      this.speed += this.friction;
    }
    if (Math.abs(this.speed) < this.friction) {
      this.speed = 0;
    }

    if (this.speed !== 0) {
      const flip = this.speed > 0 ? 1 : -1;
      if (this.controls.left) {
        this.angle += 0.03 * flip;
      }
      if (this.controls.right) {
        this.angle -= 0.03 * flip;
      }
    }

    this.x -= Math.sin(this.angle) * this.speed;
    this.y -= Math.cos(this.angle) * this.speed;
  }

  draw(ctx, color, drawSensor = false) {
    if (this.damaged) {
      ctx.fillStyle = 'gray';
    } else {
      ctx.fillStyle = color;
    }
    ctx.beginPath();
    ctx.moveTo(this.polygon[0].x, this.polygon[0].y);
    for (let i = 1; i < this.polygon.length; i++) {
      ctx.lineTo(this.polygon[i].x, this.polygon[i].y);
    }
    ctx.fill();

    if (this.sensor && drawSensor) {
      this.sensor.draw(ctx);
    }
  }
}

// Road class
class Road {
  constructor(x, width, laneCount = 3) {
    this.x = x;
    this.width = width;
    this.laneCount = laneCount;

    this.left = x - width / 2;
    this.right = x + width / 2;

    const infinity = 1000000;
    this.top = -infinity;
    this.bottom = infinity;

    const topLeft = { x: this.left, y: this.top };
    const topRight = { x: this.right, y: this.top };
    const bottomLeft = { x: this.left, y: this.bottom };
    const bottomRight = { x: this.right, y: this.bottom };
    this.borders = [
      [topLeft, bottomLeft],
      [topRight, bottomRight]
    ];

    // Create intersections
    this.intersections = [
      { y: -200, width: this.width },
      { y: -600, width: this.width },
      { y: -1000, width: this.width },
      { y: -1400, width: this.width }
    ];
  }

  getLaneCenter(laneIndex) {
    const laneWidth = this.width / this.laneCount;
    return this.left + laneWidth / 2 + Math.min(laneIndex, this.laneCount - 1) * laneWidth;
  }

  draw(ctx) {
    ctx.lineWidth = 5;
    ctx.strokeStyle = 'white';

    for (let i = 1; i <= this.laneCount - 1; i++) {
      const x = lerp(this.left, this.right, i / this.laneCount);

      ctx.setLineDash([20, 20]);
      ctx.beginPath();
      ctx.moveTo(x, this.top);
      ctx.lineTo(x, this.bottom);
      ctx.stroke();
    }

    ctx.setLineDash([]);
    this.borders.forEach(border => {
      ctx.beginPath();
      ctx.moveTo(border[0].x, border[0].y);
      ctx.lineTo(border[1].x, border[1].y);
      ctx.stroke();
    });

    // Draw intersections
    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 3;
    this.intersections.forEach(intersection => {
      const height = 80;
      ctx.fillRect(this.left, intersection.y - height / 2, this.width, height);
      ctx.strokeRect(this.left, intersection.y - height / 2, this.width, height);
      
      // Draw crosswalk lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.lineWidth = 4;
      for (let i = 0; i < 5; i++) {
        const lineX = lerp(this.left + 10, this.right - 10, i / 4);
        ctx.beginPath();
        ctx.moveTo(lineX, intersection.y - height / 2 + 5);
        ctx.lineTo(lineX, intersection.y - height / 2 + 20);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(lineX, intersection.y + height / 2 - 20);
        ctx.lineTo(lineX, intersection.y + height / 2 - 5);
        ctx.stroke();
      }
    });
  }
}

// Visualizer for neural network
class Visualizer {
  static drawNetwork(ctx, network) {
    const margin = 50;
    const left = margin;
    const top = margin;
    const width = ctx.canvas.width - margin * 2;
    const height = ctx.canvas.height - margin * 2;

    const levelHeight = height / network.levels.length;

    for (let i = network.levels.length - 1; i >= 0; i--) {
      const levelTop =
        top +
        lerp(
          height - levelHeight,
          0,
          network.levels.length === 1 ? 0.5 : i / (network.levels.length - 1)
        );

      Visualizer.drawLevel(
        ctx,
        network.levels[i],
        left,
        levelTop,
        width,
        levelHeight,
        i === network.levels.length - 1 ? ['↑', '←', '→', '↓'] : []
      );
    }
  }

  static drawLevel(ctx, level, left, top, width, height, outputLabels) {
    const right = left + width;
    const bottom = top + height;

    const { inputs, outputs, weights, biases } = level;

    for (let i = 0; i < inputs.length; i++) {
      for (let j = 0; j < outputs.length; j++) {
        ctx.beginPath();
        ctx.moveTo(Visualizer.getNodeX(inputs, i, left, right), bottom);
        ctx.lineTo(Visualizer.getNodeX(outputs, j, left, right), top);
        ctx.lineWidth = 2;
        ctx.strokeStyle = Visualizer.getRGBA(weights[i][j]);
        ctx.stroke();
      }
    }

    const nodeRadius = 18;
    for (let i = 0; i < inputs.length; i++) {
      const x = Visualizer.getNodeX(inputs, i, left, right);
      ctx.beginPath();
      ctx.arc(x, bottom, nodeRadius, 0, Math.PI * 2);
      ctx.fillStyle = 'black';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, bottom, nodeRadius * 0.6, 0, Math.PI * 2);
      ctx.fillStyle = Visualizer.getRGBA(inputs[i]);
      ctx.fill();
    }

    for (let i = 0; i < outputs.length; i++) {
      const x = Visualizer.getNodeX(outputs, i, left, right);
      ctx.beginPath();
      ctx.arc(x, top, nodeRadius, 0, Math.PI * 2);
      ctx.fillStyle = 'black';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, top, nodeRadius * 0.6, 0, Math.PI * 2);
      ctx.fillStyle = Visualizer.getRGBA(outputs[i]);
      ctx.fill();

      ctx.beginPath();
      ctx.lineWidth = 2;
      ctx.arc(x, top, nodeRadius * 0.8, 0, Math.PI * 2);
      ctx.strokeStyle = Visualizer.getRGBA(biases[i]);
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      if (outputLabels[i]) {
        ctx.beginPath();
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = 'white';
        ctx.strokeStyle = 'black';
        ctx.font = nodeRadius * 1.5 + 'px Arial';
        ctx.fillText(outputLabels[i], x, top + nodeRadius * 0.1);
        ctx.lineWidth = 0.5;
        ctx.strokeText(outputLabels[i], x, top + nodeRadius * 0.1);
      }
    }
  }

  static getNodeX(nodes, index, left, right) {
    return lerp(
      left,
      right,
      nodes.length === 1 ? 0.5 : index / (nodes.length - 1)
    );
  }

  static getRGBA(value) {
    const alpha = Math.abs(value);
    const R = value < 0 ? 0 : 255;
    const G = R;
    const B = value > 0 ? 0 : 255;
    return 'rgba(' + R + ',' + G + ',' + B + ',' + alpha + ')';
  }
}

function App() {
  const carCanvasRef = useRef(null);
  const networkCanvasRef = useRef(null);
  const [carCount, setCarCount] = useState(50);
  const [mutationRate, setMutationRate] = useState(0.2);
  const [isPaused, setIsPaused] = useState(false);
  const carsRef = useRef([]);
  const trafficRef = useRef([]);
  const trafficLightsRef = useRef([]);
  const roadRef = useRef(null);
  const bestCarRef = useRef(null);
  const animationIdRef = useRef(null);

  useEffect(() => {
    const carCanvas = carCanvasRef.current;
    const networkCanvas = networkCanvasRef.current;
    carCanvas.width = 200;
    networkCanvas.width = 300;

    const carCtx = carCanvas.getContext('2d');
    const networkCtx = networkCanvas.getContext('2d');

    roadRef.current = new Road(carCanvas.width / 2, carCanvas.width * 0.9);

    // Generate traffic lights at intersections with staggered phases
    trafficLightsRef.current = [
      new TrafficLight(-200, carCanvas.width / 2 + 60, 0),
      new TrafficLight(-600, carCanvas.width / 2 + 60, 90),
      new TrafficLight(-1000, carCanvas.width / 2 + 60, 180),
      new TrafficLight(-1400, carCanvas.width / 2 + 60, 270)
    ];

    // Generate traffic - reduced density and better spacing
    trafficRef.current = [
      new Car(roadRef.current.getLaneCenter(1), -120, 30, 50, 'DUMMY', 2),
      new Car(roadRef.current.getLaneCenter(0), -400, 30, 50, 'DUMMY', 2),
      new Car(roadRef.current.getLaneCenter(2), -700, 30, 50, 'DUMMY', 2),
      new Car(roadRef.current.getLaneCenter(1), -1100, 30, 50, 'DUMMY', 2),
      new Car(roadRef.current.getLaneCenter(0), -1300, 30, 50, 'DUMMY', 2)
    ];

    generateCars(carCount);

    animate();

    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
    };
  }, []);

  const generateCars = (N) => {
    const cars = [];
    for (let i = 0; i < N; i++) {
      // Distribute cars across lanes for better starting positions
      const lane = i % 3;
      cars.push(new Car(roadRef.current.getLaneCenter(lane), 100, 30, 50, 'AI'));
    }
    carsRef.current = cars;

    const bestBrain = localStorage.getItem('bestBrain');
    if (bestBrain) {
      for (let i = 0; i < cars.length; i++) {
        cars[i].brain = JSON.parse(bestBrain);
        if (i !== 0) {
          NeuralNetwork.mutate(cars[i].brain, mutationRate);
        }
      }
    }
  };

  const animate = (time) => {
    const carCanvas = carCanvasRef.current;
    const networkCanvas = networkCanvasRef.current;
    const carCtx = carCanvas.getContext('2d');
    const networkCtx = networkCanvas.getContext('2d');

    if (!isPaused) {
      // Update traffic lights
      for (let i = 0; i < trafficLightsRef.current.length; i++) {
        trafficLightsRef.current[i].update();
      }
      
      for (let i = 0; i < trafficRef.current.length; i++) {
        trafficRef.current[i].update(roadRef.current.borders, [], []);
      }
      for (let i = 0; i < carsRef.current.length; i++) {
        carsRef.current[i].update(roadRef.current.borders, trafficRef.current, trafficLightsRef.current);
      }
      bestCarRef.current = carsRef.current.find(
        c => c.y === Math.min(...carsRef.current.map(c => c.y))
      );
    }

    carCanvas.height = window.innerHeight;
    networkCanvas.height = window.innerHeight;

    carCtx.save();
    carCtx.translate(0, -bestCarRef.current.y + carCanvas.height * 0.7);

    roadRef.current.draw(carCtx);
    
    // Draw traffic lights
    for (let i = 0; i < trafficLightsRef.current.length; i++) {
      trafficLightsRef.current[i].draw(carCtx);
    }
    
    for (let i = 0; i < trafficRef.current.length; i++) {
      trafficRef.current[i].draw(carCtx, '#f87171');
    }
    carCtx.globalAlpha = 0.2;
    for (let i = 0; i < carsRef.current.length; i++) {
      carsRef.current[i].draw(carCtx, '#3b82f6');
    }
    carCtx.globalAlpha = 1;
    bestCarRef.current.draw(carCtx, '#22c55e', true);

    carCtx.restore();

    networkCtx.lineDashOffset = -time / 50;
    Visualizer.drawNetwork(networkCtx, bestCarRef.current.brain);
    animationIdRef.current = requestAnimationFrame(animate);
  };

  const saveBrain = () => {
    localStorage.setItem('bestBrain', JSON.stringify(bestCarRef.current.brain));
  };

  const discardBrain = () => {
    localStorage.removeItem('bestBrain');
    window.location.reload();
  };

  const handleCarCountChange = (value) => {
    setCarCount(value[0]);
  };

  const handleMutationChange = (value) => {
    setMutationRate(value[0]);
  };

  const restartSimulation = () => {
    generateCars(carCount);
  };

  return (
    <div className="app-container">
      <div className="canvas-section">
        <canvas ref={carCanvasRef} id="carCanvas" data-testid="car-canvas" />
        <canvas ref={networkCanvasRef} id="networkCanvas" data-testid="network-canvas" />
      </div>
      <div className="controls-section">
        <Card className="controls-card" data-testid="controls-panel">
          <div className="controls-header">
            <h1 className="controls-title">Self-Driving Car AI</h1>
            <p className="controls-subtitle">Neural Network Simulation</p>
          </div>

          <div className="controls-content">
            <div className="control-group">
              <label className="control-label">
                AI Cars: <span className="control-value">{carCount}</span>
              </label>
              <Slider
                data-testid="car-count-slider"
                value={[carCount]}
                onValueChange={handleCarCountChange}
                min={1}
                max={200}
                step={1}
                className="control-slider"
              />
            </div>

            <div className="control-group">
              <label className="control-label">
                Mutation Rate: <span className="control-value">{(mutationRate * 100).toFixed(0)}%</span>
              </label>
              <Slider
                data-testid="mutation-slider"
                value={[mutationRate]}
                onValueChange={handleMutationChange}
                min={0}
                max={1}
                step={0.01}
                className="control-slider"
              />
            </div>

            <div className="button-group">
              <Button
                data-testid="save-brain-btn"
                onClick={saveBrain}
                className="action-button save-button"
              >
                <Save className="button-icon" />
                Save Best Brain
              </Button>
              <Button
                data-testid="discard-brain-btn"
                onClick={discardBrain}
                variant="destructive"
                className="action-button"
              >
                <Trash2 className="button-icon" />
                Discard Brain
              </Button>
              <Button
                data-testid="restart-btn"
                onClick={restartSimulation}
                variant="outline"
                className="action-button"
              >
                <Play className="button-icon" />
                Restart
              </Button>
            </div>

            <div className="info-section">
              <div className="info-item">
                <div className="sensor-indicator green" />
                <span>Active Sensors</span>
              </div>
              <div className="info-item">
                <div className="sensor-indicator red" />
                <span>Obstacle Detected</span>
              </div>
              <div className="info-item">
                <div className="car-indicator best" />
                <span>Best Performing AI</span>
              </div>
              <div className="info-item">
                <div className="car-indicator traffic" />
                <span>Traffic Obstacles</span>
              </div>
              <div className="info-item">
                <div className="traffic-light-indicator" />
                <span>Traffic Lights</span>
              </div>
            </div>

            <div className="instructions">
              <h3 className="instructions-title">How it works:</h3>
              <ul className="instructions-list">
                <li>Cars use ray-casting sensors to detect obstacles</li>
                <li>Neural network controls steering, acceleration & braking</li>
                <li>AI learns to obey traffic lights at intersections</li>
                <li>Best performing car is highlighted in green</li>
                <li>Save the brain when a car performs well</li>
                <li>Mutation creates variations for learning</li>
              </ul>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

export default App;
