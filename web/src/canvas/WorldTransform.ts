/** Coordinate transform between world (image) and screen (canvas) space. */

export class WorldTransformImpl {
  constructor(
    public scale: number,
    public offsetX: number,
    public offsetY: number,
  ) {}

  /** Convert world coordinates to screen (canvas pixel) coordinates. */
  toScreen(wx: number, wy: number): [sx: number, sy: number] {
    return [wx * this.scale + this.offsetX, wy * this.scale + this.offsetY];
  }

  /** Convert screen coordinates to world (image pixel) coordinates. */
  toWorld(sx: number, sy: number): [wx: number, wy: number] {
    return [
      (sx - this.offsetX) / this.scale,
      (sy - this.offsetY) / this.scale,
    ];
  }

  /** Scale a distance from world to screen. */
  scaleDistance(d: number): number {
    return d * this.scale;
  }

  /** Scale a distance from screen to world. */
  unscaleDistance(d: number): number {
    return d / this.scale;
  }
}

/** Create a WorldTransform from the current canvas store state. */
export function createTransform(
  zoom: number,
  panX: number,
  panY: number,
): WorldTransformImpl {
  return new WorldTransformImpl(zoom, panX, panY);
}
