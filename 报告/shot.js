/**
 * 用 Chrome DevTools Protocol 驱动真实浏览器，截图并驱动真实对话。
 * 运行时：node shot.js
 */
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");
const http = require("http");

const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const DBG_PORT = 9333;
const APP = "http://127.0.0.1:5099";
const OUT = path.join(__dirname, "screenshots");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function getJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let data = "";
      res.on("data", (c) => (data += c));
      res.on("end", () => {
        try { resolve(JSON.parse(data)); } catch (e) { reject(e); }
      });
    }).on("error", reject);
  });
}

class CDP {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
      }
    });
  }
  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`CDP 超时: ${method}`));
        }
      }, 90000);
    });
  }
  async eval(expression) {
    const r = await this.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error("JS 异常: " + JSON.stringify(r.exceptionDetails).slice(0, 400));
    return r.result.value;
  }
}

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "cdp-profile-"));
  const child = spawn(
    CHROME,
    [
      "--headless=new",
      `--remote-debugging-port=${DBG_PORT}`,
      `--user-data-dir=${profile}`,
      "--window-size=1500,1700",
      "--hide-scrollbars",
      "--force-device-scale-factor=1",
      "--force-color-profile=srgb",
      "--no-first-run",
      "--no-default-browser-check",
      "about:blank",
    ],
    { stdio: "ignore" }
  );

  let target = null;
  for (let i = 0; i < 60; i++) {
    try {
      const list = await getJson(`http://127.0.0.1:${DBG_PORT}/json/list`);
      const page = list.find((t) => t.type === "page");
      if (page && page.webSocketDebuggerUrl) { target = page; break; }
    } catch (e) { /* 还没起来 */ }
    await sleep(500);
  }
  if (!target) throw new Error("Chrome 调试端口未就绪");

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => {
    ws.addEventListener("open", res);
    ws.addEventListener("error", rej);
  });
  const cdp = new CDP(ws);
  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");

  async function viewport(width, height) {
    await cdp.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
  }
  async function shot(name) {
    const r = await cdp.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
    fs.writeFileSync(path.join(OUT, name), Buffer.from(r.data, "base64"));
    console.log("shot:", name);
  }
  async function goto(url) {
    await cdp.send("Page.navigate", { url });
    await sleep(1500);
  }
  async function ask(phone, question, shotName, { expandTools = false, maxWaitMs = 120000 } = {}) {
    await cdp.eval(`(() => {
      const p = document.getElementById("phone"); p.value = ${JSON.stringify(phone)};
      p.dispatchEvent(new Event("input", {bubbles:true}));
      const q = document.getElementById("query");
      q.value = ${JSON.stringify(question)};
      q.dispatchEvent(new Event("input", {bubbles:true}));
      document.getElementById("sendBtn").click();
      return true;
    })()`);
    const start = Date.now();
    let last = "";
    while (Date.now() - start < maxWaitMs) {
      await sleep(1500);
      const txt = await cdp.eval(`document.getElementById("chatBox").innerText`);
      if (txt === last && txt.length > 0) {
        const busy = await cdp.eval(`document.getElementById("sendBtn").disabled`);
        if (!busy) break;
      }
      last = txt;
    }
    if (expandTools) {
      await cdp.eval(`(() => {
        const msgs = document.querySelectorAll(".msg.assistant");
        const el = msgs[msgs.length - 1];
        const d = el.querySelector("details.tools");
        if (d) d.open = true;
        document.getElementById("chatBox").scrollTop = document.getElementById("chatBox").scrollHeight;
        return !!d;
      })()`);
    }
    await sleep(700);
    await shot(shotName);
  }

  // 1) 首页（常规窗口尺寸）
  await viewport(1440, 900);
  await goto(APP + "/");
  await sleep(1500);
  await shot("01-首页.png");

  // 2) 新用户欢迎语
  await viewport(1500, 1650);
  await ask("13700002222", "你好，我想了解一下你们的课程", "02-新用户欢迎语.png");

  // 3) 课程咨询（意图 2）+ 展开工具调用
  await ask("13700002222", "你们有哪些 Python 课程？多少钱？", "03-课程咨询意图2.png", { expandTools: true });

  // 4) 职业咨询（意图 1）+ 展开工具调用
  await ask("13700002222", "AI 应用开发工程师现在薪资多少？", "04-职业咨询意图1.png", { expandTools: true });

  // 5) 运维接口 JSON 视图
  await viewport(1440, 900);
  await goto(APP + "/vectorstore/list");
  await shot("05-向量库列表接口.png");

  await goto(APP + "/database/schema");
  await shot("06-数据库结构接口.png");

  await goto(APP + "/health");
  await shot("07-健康检查接口.png");

  ws.close();
  child.kill();
  console.log("DONE");
}

main().catch((e) => { console.error("FAILED:", e); process.exit(1); });
