import { defineConfig } from 'vite';

export default defineConfig({
  // 从仓库根目录读取 Vite 环境文件，页面入口仍放在 src/app。
  envDir: '../..',
  root: 'src/app',
  // GitHub Pages 项目页挂在 /prism/ 下；本地 dev 和普通静态托管仍在根。
  // 只有 Actions 里构建时才加前缀，本地 npm run build 的产物仍然可直接预览。
  base: process.env.GITHUB_ACTIONS ? '/prism/' : '/',
  // 资产（模型 / 贴图 / 环境图 / 画廊封面）原样发布到站点根目录
  publicDir: '../../assets',
  server: {
    port: 5173,
    open: true,
  },
  build: {
    outDir: '../../dist/app',
    // 输出目录在项目根之外（dist/app），vite 默认不敢清空，旧 chunk 会一直堆着
    emptyOutDir: true,
  },
  // 渲染 Worker 和主线程共用 core / engine 模块，会拆出共享 chunk ——
  // 只支持 ES 格式（默认的 iife 不能 code-split）
  worker: {
    format: 'es',
  },
});
