/**
 * 浏览器级 UI 回归测试：用 jsdom 加载 **Flask 实际返回的页面**（http://127.0.0.1:5000/），
 * 真实执行页面内联 JS，验证：
 *   1. 服务端下发的 HTML 没有被二次转义（\\\\d / \\\\n 这类双反斜杠 bug）；
 *   2. 手机号校验逻辑（合法号通过、非法号弹窗）；
 *   3. 点击发送后 SSE 流式事件被正确解析并渲染（此前 bug 的核心）；
 *   4. 多轮对话、意图/工具调用/画像标签渲染、快捷按钮、Enter 发送、健康检查。
 *
 * 运行：node ui_smoke.js
 * 依赖：jsdom（仅开发自测用，不属于 Flask 运行依赖）：npm install jsdom
 * 环境变量：CRM_BASE_URL 可覆盖服务地址（默认 http://127.0.0.1:5000）。
 */
const { JSDOM } = require("jsdom");

const BASE = (process.env.CRM_BASE_URL || "http://127.0.0.1:5000").replace(/\/$/, "");

const failed = [];
const passed = [];
function check(name, cond, detail) {
  if (cond) { passed.push(name); console.log(`PASS  ${name}`); }
  else { failed.push(name); console.log(`FAIL  ${name}${detail ? " | " + detail : ""}`); }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const resp = await fetch(BASE + "/");
  check("GET / 返回 200", resp.status === 200, `status=${resp.status}`);
  const html = await resp.text();
  check("服务端页面未出现双反斜杠正则（\\\\d）", !html.includes("\\\\d{9}"));
  check("服务端页面未出现双反斜杠换行（\\\\n\\\\n）", !html.includes("\\\\n\\\\n"));
  check("服务端页面包含正确的 SSE 分帧代码", html.includes('split("\\n\\n")'));

  const alerts = [];
  const dom = new JSDOM(html, {
    url: BASE + "/",
    runScripts: "dangerously",
    pretendToBeVisual: true,
    beforeParse(window) {
      window.alert = (msg) => alerts.push(String(msg));
      window.fetch = (input, init) => fetch(new URL(input, BASE + "/"), init);
      window.TextDecoder = TextDecoder;
      window.TextEncoder = TextEncoder;
    },
  });
  const { window } = dom;
  const doc = window.document;

  const phone = () => doc.getElementById("phone");
  const query = () => doc.getElementById("query");
  const sendBtn = () => doc.getElementById("sendBtn");
  const chatBox = () => doc.getElementById("chatBox");
  const userMsgs = () => chatBox().querySelectorAll(".msg.user");
  const assistantMsgs = () => chatBox().querySelectorAll(".msg.assistant");
  const lastAssistant = () => { const arr = assistantMsgs(); return arr[arr.length - 1]; };

  async function waitFor(fn, timeout = 90000, interval = 150) {
    const start = Date.now();
    while (Date.now() - start < timeout) {
      try { if (fn()) return true; } catch (_) { /* ignore */ }
      await sleep(interval);
    }
    return false;
  }
  const replyText = (el) => ((el && el.querySelector(".text") && el.querySelector(".text").textContent) || "").trim();
  const metaText = (el) => { const m = el && el.querySelector(".extra"); return m ? m.textContent : ""; };

  await sleep(1200); // 等待页面初始化脚本执行完

  // ---------- 0. 页面基础渲染 ----------
  check("页面标题渲染", doc.querySelector("header h1").textContent.includes("SeaChat"));
  check("顶部技术栈徽标已移除", !doc.querySelector("header .badge"));
  check("初始欢迎消息存在", assistantMsgs().length === 1 && replyText(lastAssistant()).includes("SeaChat"));
  check("手机号默认值可被校验为合法", /^1[3-9]\d{9}$/.test(phone().value.trim()), phone().value);
  check("初始无 alert", alerts.length === 0, JSON.stringify(alerts));

  // ---------- 1. 非法手机号被拦截 ----------
  phone().value = "123456";
  query().value = "你好";
  sendBtn().click();
  await sleep(300);
  check("非法手机号弹窗拦截", alerts.length === 1 && alerts[0].includes("11 位手机号"), JSON.stringify(alerts));
  check("非法手机号未发送（无 user 消息）", userMsgs().length === 0);
  check("非法手机号时输入框内容保留", query().value === "你好");

  // ---------- 2. 合法手机号首轮：应拿到欢迎语（SSE 流式解析） ----------
  const testPhone = "13" + String(Date.now()).slice(-9); // 每次运行使用新号码，保证首轮必是欢迎语
  phone().value = testPhone;
  query().value = "你好";
  sendBtn().click();
  // 关键：等「本轮结束」——done 事件渲染完 meta 且 sendBtn 恢复可用（finally 里 busy=false / disabled=false）。
  // 真实 LLM 模式单轮可能十几秒，之前脚本只等 delta 就点下一轮，会被 busy 拦截，属于测试自身竞态。
  const firstOk = await waitFor(
    () => userMsgs().length === 1 && replyText(lastAssistant()).length > 10 && !sendBtn().disabled,
    180000
  );
  check("首轮发送后渲染出回答（SSE 解析正常）", firstOk, `text=${replyText(lastAssistant()).slice(0, 60)}`);
  const firstReply = replyText(lastAssistant());
  check("首轮回答是欢迎语（新用户规范）", firstReply.includes("欢迎"), firstReply);
  check("首轮未残留思考占位", !firstReply.includes("正在思考"));
  check("合法手机号不再弹窗", alerts.length === 1, JSON.stringify(alerts));
  check("首轮 user 消息回流", userMsgs().length === 1 && userMsgs()[0].textContent.includes("你好"));
  check("首轮输入框已清空", query().value === "");

  // ---------- 3. 第二轮（同一手机号）：课程咨询 → 意图 2 + 工具调用 ----------
  query().value = "你们有哪些 Python 课程？多少钱？";
  sendBtn().click();
  const secondOk = await waitFor(() => {
    const el = lastAssistant();
    return (
      userMsgs().length === 2 &&
      metaText(el).includes("意图") &&
      replyText(el).length > 20 &&
      !sendBtn().disabled
    );
  }, 180000);
  const secondReply = replyText(lastAssistant());
  const secondMeta = metaText(lastAssistant());
  check("第二轮渲染正常且带意图标签", secondOk, secondMeta);
  check("第二轮意图识别为课程咨询(2)", secondMeta.includes("意图 2"), secondMeta);
  check("第二轮展示工具调用标签", secondMeta.includes("工具调用"), secondMeta);
  check(
    "工具调用下拉框独立渲染（不再挤在标签行）",
    !lastAssistant().querySelector(".meta details") && !!lastAssistant().querySelector(".extra > details.tools")
  );
  check("第二轮回答含课程信息", secondReply.includes("课程") && (secondReply.includes("元") || secondReply.includes("9800")), secondReply.slice(0, 120));
  check("模型答案中的 Markdown 已渲染（不再出现 ** 星号）", !secondReply.includes("**"), secondReply.slice(0, 120));
  check("流式增量无残留思考占位", !secondReply.includes("正在思考"));

  // ---------- 4. 快捷按钮与 Enter 发送 ----------
  const quick = doc.querySelectorAll(".quick button");
  quick[2].click();
  check("快捷按钮写入输入框", query().value.includes("天气"), query().value);
  query().value = "AI 应用开发工程师薪资前景怎么样？";
  query().dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true }));
  const thirdOk = await waitFor(
    () => userMsgs().length === 3 && metaText(lastAssistant()).includes("意图 1") && !sendBtn().disabled,
    240000
  );
  check("Enter 发送并识别为职业咨询(1)", thirdOk, metaText(lastAssistant()));
  const thirdMeta = metaText(lastAssistant());
  check("职业咨询展示工具调用", thirdMeta.includes("工具调用"), thirdMeta);
  check("职业咨询回答含薪资信息", replyText(lastAssistant()).includes("薪") || replyText(lastAssistant()).includes("元"), replyText(lastAssistant()).slice(0, 120));

  // ---------- 5. 健康检查状态灯 ----------
  const llmState = doc.getElementById("llmState");
  check("健康检查状态已更新（非检测中）", !llmState.textContent.includes("检测中"), llmState.textContent);

  console.log(`\n结果：${passed.length} passed, ${failed.length} failed`);
  if (failed.length) { console.log("失败项：" + failed.join("；")); process.exit(1); }
  process.exit(0);
}

main().catch((err) => {
  console.error("测试执行异常：", err);
  process.exit(2);
});
