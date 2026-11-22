import type { CameraHandle } from '../core/api/prism';

type DragMode = 'orbit' | 'pan' | 'zoom';

/** 指针手感系数：每像素转多少弧度、推拉多快。属于界面，不属于相机语义。 */
const ORBIT_PER_PIXEL = 0.008;
const DRAG_ZOOM_PER_PIXEL = 0.005;
const WHEEL_ZOOM_IN = 0.9;
const WHEEL_ZOOM_OUT = 1.1;

/**
 * 画布上的轨道操作：鼠标拖动、滚轮、触屏。
 *
 * 鼠标映射照 Blender 的视口：
 *
 *   中键拖       转视角
 *   Shift + 中键 平移
 *   Ctrl/Cmd + 中键  推拉
 *   滚轮         推拉
 *   右键拖       平移
 *
 * 左键也留给旋转：Mac 触控板没有中键，不然只剩右键平移、没法转视角了。
 * 触屏单指转，双指捏合推拉。
 *
 * 这里只产出台阶式的相机命令（orbit / pan / zoom），不碰渲染和 DOM 结构。
 */
export class OrbitControls {
    private camera: CameraHandle | null = null;
    private target: HTMLElement | null = null;
    private mode: DragMode | null = null;
    private lastX = 0;
    private lastY = 0;
    private pinchDistance = 0;
    private readonly onMouseDown = (e: MouseEvent) => {
        const mode = mouseMode(e);
        if (!mode) return;
        this.mode = mode;
        this.lastX = e.clientX;
        this.lastY = e.clientY;
        e.preventDefault();
    };
    private readonly onMouseMove = (e: MouseEvent) => {
        if (!this.mode) return;
        const dx = e.clientX - this.lastX;
        const dy = e.clientY - this.lastY;
        this.lastX = e.clientX;
        this.lastY = e.clientY;
        if (this.mode === 'orbit') {
            this.camera?.orbit(dx * ORBIT_PER_PIXEL, -dy * ORBIT_PER_PIXEL);
        } else if (this.mode === 'pan') {
            this.camera?.pan(dx, dy);
        } else {
            // 往上拖推近：指数映射，拖多远都是等比例，不会突然冲过头
            this.camera?.zoom(Math.exp(dy * DRAG_ZOOM_PER_PIXEL));
        }
    };
    private readonly onMouseUp = () => {
        this.mode = null;
    };
    private readonly onWheel = (e: WheelEvent) => {
        e.preventDefault();
        this.camera?.zoom(e.deltaY > 0 ? WHEEL_ZOOM_OUT : WHEEL_ZOOM_IN);
    };
    private readonly onContextMenu = (e: Event) => {
        e.preventDefault();
    };
    private readonly onTouchStart = (e: TouchEvent) => {
        if (e.touches.length >= 2) {
            this.mode = null;
            this.pinchDistance = touchDistance(e);
            return;
        }
        if (e.touches.length !== 1) return;
        this.pinchDistance = 0;
        this.mode = 'orbit';
        this.lastX = e.touches[0].clientX;
        this.lastY = e.touches[0].clientY;
    };
    private readonly onTouchMove = (e: TouchEvent) => {
        e.preventDefault();
        if (this.pinchDistance > 0 && e.touches.length >= 2) {
            const next = touchDistance(e);
            if (next > 0) this.camera?.zoom(this.pinchDistance / next);
            this.pinchDistance = next;
            return;
        }
        if (this.mode !== 'orbit' || e.touches.length !== 1) return;
        const touch = e.touches[0];
        const dx = touch.clientX - this.lastX;
        const dy = touch.clientY - this.lastY;
        this.lastX = touch.clientX;
        this.lastY = touch.clientY;
        this.camera?.orbit(dx * ORBIT_PER_PIXEL, -dy * ORBIT_PER_PIXEL);
    };
    private readonly onTouchEnd = () => {
        this.mode = null;
        this.pinchDistance = 0;
    };

    public attach(target: HTMLElement, camera: CameraHandle): void {
        this.detach();
        this.target = target;
        this.camera = camera;
        target.addEventListener('mousedown', this.onMouseDown);
        target.addEventListener('wheel', this.onWheel, { passive: false });
        target.addEventListener('contextmenu', this.onContextMenu);
        target.addEventListener('touchstart', this.onTouchStart, { passive: false });
        target.addEventListener('touchmove', this.onTouchMove, { passive: false });
        target.addEventListener('touchend', this.onTouchEnd);
        window.addEventListener('mousemove', this.onMouseMove);
        window.addEventListener('mouseup', this.onMouseUp);
    }

    public detach(): void {
        const target = this.target;
        if (target) {
            target.removeEventListener('mousedown', this.onMouseDown);
            target.removeEventListener('wheel', this.onWheel);
            target.removeEventListener('contextmenu', this.onContextMenu);
            target.removeEventListener('touchstart', this.onTouchStart);
            target.removeEventListener('touchmove', this.onTouchMove);
            target.removeEventListener('touchend', this.onTouchEnd);
        }
        window.removeEventListener('mousemove', this.onMouseMove);
        window.removeEventListener('mouseup', this.onMouseUp);
        this.target = null;
        this.camera = null;
        this.mode = null;
        this.pinchDistance = 0;
    }
}

function mouseMode(e: MouseEvent): DragMode | null {
    if (e.button === 0) return e.shiftKey ? 'pan' : 'orbit';
    if (e.button === 1) return e.ctrlKey || e.metaKey ? 'zoom' : e.shiftKey ? 'pan' : 'orbit';
    if (e.button === 2) return 'pan';
    return null;
}

function touchDistance(e: TouchEvent): number {
    const a = e.touches[0];
    const b = e.touches[1];
    return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
}
