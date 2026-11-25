-- ===========================================================================
-- AI CRM 数据库结构定义（SQLite，UTF-8 编码，幂等执行）
-- 说明：
--   1. 本文件由 database/db.py::init_db() 执行，所有建表语句均带 IF NOT EXISTS，
--      因此可以重复执行而不会报错；
--   2. 分界标记（SEED_DATA）之前的 DDL 每次初始化都会执行，之后的示例数据
--      只在 courses / career_info 为空时执行（INSERT OR IGNORE 进一步保证幂等）；
--   3. 4 张业务表：users（用户画像）、chat_history（聊天记录）、
--      courses（课程）、career_info（职业信息）。
-- ===========================================================================

-- 用户画像表：手机号作为对外主键，记录咨询意向、预算与所处阶段
CREATE TABLE IF NOT EXISTS users (
    phone_number        TEXT PRIMARY KEY,                              -- 手机号（业务主键）
    name                TEXT,                                          -- 姓名/称呼
    intention_direction TEXT,                                          -- 意向方向，如 AI 职业、Python 课程
    budget              REAL,                                          -- 预算金额（单位：元）
    stage               TEXT NOT NULL DEFAULT '新咨询'
                        CHECK (stage IN ('新咨询', '已了解', '已报名')), -- 咨询阶段
    notes               TEXT,                                          -- 备注/用户关注点
    created_at          TEXT NOT NULL DEFAULT (datetime('now', 'localtime')), -- 创建时间
    updated_at          TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))  -- 最后更新时间
);

-- 聊天历史表：按 id 自增保存每一条 user / assistant 消息
CREATE TABLE IF NOT EXISTS chat_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,                    -- 自增主键（天然的时间顺序）
    phone_number TEXT NOT NULL,                                        -- 用户手机号
    role         TEXT NOT NULL CHECK (role IN ('user', 'assistant')),  -- 消息角色
    content      TEXT NOT NULL,                                        -- 消息内容
    intent_code  INTEGER,                                              -- 本轮意图编号 0/1/2（可为空）
    created_at   TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))  -- 创建时间
);

-- 课程表：供课程咨询、SQL 查询与向量库示例复用
CREATE TABLE IF NOT EXISTS courses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,                     -- 自增主键
    name        TEXT NOT NULL UNIQUE,                                  -- 课程名称（中文，语义检索可命中）
    category    TEXT NOT NULL,                                         -- 课程分类，如 人工智能 / 数据分析
    price       REAL NOT NULL,                                         -- 课程价格（单位：元）
    duration    TEXT NOT NULL,                                         -- 学习周期，如 "24 周"
    level       TEXT,                                                  -- 难度/基础要求
    suitable_for TEXT,                                                 -- 适合人群
    description TEXT NOT NULL,                                         -- 课程简介
    created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))   -- 创建时间
);

-- 职业信息表：岗位、行业、月薪区间与技能要求，供职业咨询与薪资查询
CREATE TABLE IF NOT EXISTS career_info (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,                     -- 自增主键
    title       TEXT NOT NULL UNIQUE,                                  -- 岗位名称
    industry    TEXT NOT NULL,                                         -- 所属行业
    salary_min  INTEGER NOT NULL,                                      -- 月薪下限（单位：元）
    salary_max  INTEGER NOT NULL,                                      -- 月薪上限（单位：元）
    skills      TEXT,                                                  -- 核心技能要求
    description TEXT,                                                  -- 岗位职责/说明
    created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))   -- 创建时间
);

-- 常用查询索引：按手机号查历史、按分类/价格查课程、按薪资查职业
CREATE INDEX IF NOT EXISTS idx_chat_history_phone     ON chat_history (phone_number);
CREATE INDEX IF NOT EXISTS idx_chat_history_phone_id  ON chat_history (phone_number, id);
CREATE INDEX IF NOT EXISTS idx_courses_category       ON courses (category);
CREATE INDEX IF NOT EXISTS idx_courses_price          ON courses (price);
CREATE INDEX IF NOT EXISTS idx_career_salary_max      ON career_info (salary_max);
CREATE INDEX IF NOT EXISTS idx_users_intention        ON users (intention_direction);

-- @@SEED_DATA@@
-- 以下为演示数据：仅在 courses / career_info 为空时由 init_db() 执行；
-- 固定 id + INSERT OR IGNORE，保证重复初始化不会产生重复数据。
INSERT OR IGNORE INTO courses
    (id, name, category, price, duration, level, suitable_for, description)
VALUES
    (1, 'Python 全栈开发实战', '编程开发', 12800.0, '24 周', '零基础',
     '零基础转行、在校学生',
     '从 Python 语法、数据结构到 Flask/Django 后端、MySQL 数据库与项目部署，完整覆盖全栈开发链路，配套 3 个企业级项目。'),
    (2, 'Python 数据分析与可视化', '数据分析', 9800.0, '16 周', '有 Python 基础',
     '转行数据分析、业务运营',
     '系统学习 NumPy、Pandas、Matplotlib、SQL 与 BI 工具，完成电商与用户增长两个数据分析项目，掌握指标体系与报表搭建。'),
    (3, 'AI 应用开发工程师（大模型 RAG）', '人工智能', 15800.0, '20 周', '会 Python 基础',
     '想转型 AI 应用开发的后端/开发人员',
     '基于大模型 API、LangChain、向量数据库与 RAG 技术，实战搭建企业知识库问答、智能客服与 AI Agent，掌握提示词工程与项目上线。'),
    (4, '大模型工程与微调实战', '人工智能', 19800.0, '24 周', '有后端或算法基础',
     '算法工程师、AI 方向进阶者',
     '覆盖 Transformer 原理、大模型微调（LoRA/QLoRA）、推理优化与私有化部署，完成一个行业大模型微调项目。'),
    (5, '前端开发实战（Vue3 + TypeScript）', '前端开发', 8800.0, '16 周', '零基础',
     '零基础、想快速就业的学员',
     '从 HTML/CSS/JavaScript 到 Vue3、TypeScript、工程化与小程序开发，完成 2 个可上线的前端项目。'),
    (6, '算法工程师进阶（机器学习方向）', '算法', 16800.0, '20 周', '有数学与 Python 基础',
     '算法岗求职者、计算机专业学生',
     '系统学习机器学习、深度学习、推荐系统与竞赛实战，掌握特征工程、模型调优与论文复现能力。'),
    (7, 'Python 自动化办公训练营', '办公自动化', 3980.0, '8 周', '零基础',
     '职场白领、行政财务',
     '用 Python 处理 Excel、Word、邮件与网页数据，掌握自动化报表和批量处理脚本，提升日常办公效率。');

INSERT OR IGNORE INTO career_info
    (id, title, industry, salary_min, salary_max, skills, description)
VALUES
    (1, 'AI 应用开发工程师', '互联网/人工智能', 15000, 28000,
     'Python、大模型 API、LangChain、RAG、Prompt Engineering、工程化部署',
     '负责把大模型能力落地到业务系统，搭建知识库问答、智能客服与 Agent 应用，是当前招聘需求增长最快的 AI 岗位之一。'),
    (2, '大模型算法工程师', '人工智能', 25000, 50000,
     '深度学习、Transformer、PyTorch、微调、推理优化',
     '负责大模型训练、微调与推理优化，要求扎实的算法与工程能力，薪资处于 AI 岗位顶端。'),
    (3, '数据分析师', '互联网/金融', 12000, 25000,
     'SQL、Python、Pandas、数据可视化、业务分析',
     '负责业务数据分析、指标体系搭建与决策支持，SQL 与 Python 是核心技能，转行门槛相对友好。'),
    (4, 'Python 后端开发工程师', '互联网', 13000, 26000,
     'Python、Flask、Django、MySQL、Redis、Docker',
     '负责服务端接口与系统开发，Python 生态成熟，适合作为转行后的第一份开发岗。'),
    (5, '算法工程师', '人工智能/互联网', 20000, 40000,
     '机器学习、深度学习、推荐系统、特征工程',
     '负责推荐、搜索、风控等算法模型的设计与迭代，岗位需求稳定，对数学与工程能力要求较高。'),
    (6, 'AI 产品经理', '人工智能/互联网', 18000, 35000,
     '需求分析、大模型应用、数据分析、用户研究',
     '负责 AI 产品规划与落地，需要理解大模型能力边界并协调研发资源，适合有产品经验的转型者。'),
    (7, 'Python 初级开发工程师', '互联网/软件服务', 8000, 15000,
     'Python 基础、Flask、MySQL、Git',
     '面向零基础转行的入门岗位，主要负责功能模块开发与维护，是进入开发行业的起始岗位。'),
    (8, '数据运营专员', '互联网/电商', 9000, 16000,
     'Excel、SQL、数据分析、报表',
     '负责业务数据监控、报表输出与运营活动复盘，适合作为转行数据方向的过渡岗位。');
