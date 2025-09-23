// App.tsx — React + TypeScript + Tailwind single-file frontend (Markdown + Typewriter)
// Drop into a Vite React TS project (src/App.tsx). Requires:
//   npm i react-markdown remark-gfm

import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// ====== Types aligned to your FastAPI schema ======

type Role = "user" | "assistant" | "system";

interface DocPreview {
  metadata: Record<string, any>;
  preview: string;
}

interface ChatResponse {
  session_id: string;
  answer: string;
  route?: string | null;
  used_docs: DocPreview[];
}

interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  route?: string | null;
  used_docs?: DocPreview[];
}

// ====== Utilities ======
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const SESSION_KEY = "copyright_chat_session_id";

function cls(...parts: (string | false | undefined | null)[]) {
  return parts.filter(Boolean).join(" ");
}

function getDocTitle(meta: Record<string, any>): string {
  return (
    meta?.doc_title ||
    meta?.title ||
    meta?.source ||
    meta?.doc_id ||
    meta?.case_id ||
    "출처 미상"
  );
}

function formatMeta(meta: Record<string, any>): string {
  const keysPreferred = [
    "doc_title",
    "case_id",
    "source",
    "doc_id",
    "year",
    "court",
    "page",
  ];
  const items: string[] = [];
  for (const k of keysPreferred) if (meta?.[k]) items.push(`${k}: ${meta[k]}`);
  const rest = Object.entries(meta || {})
    .filter(([k]) => !keysPreferred.includes(k))
    .map(([k, v]) => `${k}: ${v}`);
  return [...items, ...rest].join(" · ");
}

function useLocalSessionId() {
  const [sid, setSid] = useState<string | null>(() =>
    localStorage.getItem(SESSION_KEY)
  );
  const update = (next: string) => {
    localStorage.setItem(SESSION_KEY, next);
    setSid(next);
  };
  const clear = () => {
    localStorage.removeItem(SESSION_KEY);
    setSid(null);
  };
  return { sid, update, clear } as const;
}

// ====== API ======
async function postChat(q: string, sessionId?: string | null) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(sessionId ? { "X-Session-Id": sessionId } : {}),
    },
    body: JSON.stringify({ q }),
  });
  const sid = res.headers.get("X-Session-Id");
  if (!res.ok) throw new Error(await res.text());
  const data = (await res.json()) as ChatResponse;
  return { data, returnedSessionId: sid };
}

// ====== Markdown renderer ======
const Md: React.FC<{ children: string }> = ({ children }) => (
  <div className="markdown text-[14px] leading-relaxed">
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ node, ...props }) => (
          <a
            {...props}
            target="_blank"
            rel="noreferrer"
            className="text-indigo-600 underline"
          />
        ),
        code: ({ inline, className, children, ...props }) => (
          <code
            className={cls(
              "rounded bg-gray-100 px-1 py-0.5 font-mono text-[12px]",
              !inline && "block whitespace-pre-wrap break-words p-2"
            )}
            {...props}
          >
            {children}
          </code>
        ),
        ul: (props) => <ul className="list-disc pl-5" {...props} />,
        ol: (props) => <ol className="list-decimal pl-5" {...props} />,
        h1: (props) => <h1 className="mb-2 text-lg font-semibold" {...props} />,
        h2: (props) => (
          <h2 className="mb-2 text-base font-semibold" {...props} />
        ),
        blockquote: (props) => (
          <blockquote
            className="border-l-4 border-gray-300 pl-3 text-gray-700"
            {...props}
          />
        ),
      }}
    >
      {children}
    </ReactMarkdown>
  </div>
);

// ====== Typewriter effect (client-side incremental rendering) ======
function useTypewriter(full: string, enabled: boolean, speedMs = 14) {
  const [partial, setPartial] = useState(enabled ? "" : full);
  useEffect(() => {
    if (!enabled) {
      setPartial(full);
      return;
    }
    setPartial("");
    let i = 0;
    let cancelled = false;
    const tick = () => {
      if (cancelled) return;
      i = Math.min(i + 3, full.length); // add characters in small groups for speed
      setPartial(full.slice(0, i));
      if (i < full.length) setTimeout(tick, speedMs);
    };
    tick();
    return () => {
      cancelled = true;
    };
  }, [full, enabled, speedMs]);
  return partial;
}

// ====== UI Components ======
function Header({ onNew }: { onNew: () => void }) {
  return (
    <header className="sticky top-0 z-20 border-b bg-white/80 backdrop-blur supports-[backdrop-filter]:bg-white/60">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-xl bg-indigo-600" />
          <div>
            <h1 className="text-lg font-semibold">저작권법 자문 챗봇</h1>
            <p className="text-xs text-gray-500">RAG · pgvector · LangChain</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <a
            className="text-sm text-gray-500 hover:text-gray-700"
            href="/docs"
            target="_blank"
            rel="noreferrer"
          >
            Swagger
          </a>
          <button
            onClick={onNew}
            className="rounded-xl border px-3 py-1.5 text-sm font-medium hover:bg-gray-50"
          >
            New Chat
          </button>
        </div>
      </div>
    </header>
  );
}

function SourceCard({ doc, index }: { doc: DocPreview; index: number }) {
  const [open, setOpen] = useState(false);
  const title = getDocTitle(doc.metadata);
  const metaLine = formatMeta(doc.metadata);
  return (
    <div className="rounded-xl border bg-gray-50">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-[13px]"
      >
        <span className="truncate font-medium">
          [{index}] {title}
          <span className="ml-2 truncate text-gray-500">
            {metaLine && `— ${metaLine}`}
          </span>
        </span>
        <span className="text-xs text-gray-500">{open ? "Hide" : "Show"}</span>
      </button>
      {open && (
        <div className="border-t bg-white p-3 text-[13px] leading-relaxed">
          <pre className="whitespace-pre-wrap break-words text-gray-800">
            {doc.preview}
          </pre>
        </div>
      )}
    </div>
  );
}

function MessageBubble({
  m,
  typewriter,
}: {
  m: ChatMessage;
  typewriter: boolean;
}) {
  const isUser = m.role === "user";
  const rendered = useTypewriter(m.content, typewriter && !isUser);
  return (
    <div className={cls("flex w-full gap-3", isUser && "justify-end")}>
      {!isUser && (
        <div className="mt-1 h-8 w-8 flex-none rounded-full bg-indigo-600" />
      )}
      <div
        className={cls(
          "max-w-[85%] rounded-2xl border px-4 py-3 text-sm",
          isUser ? "bg-indigo-50 border-indigo-100" : "bg-white"
        )}
      >
        {m.route && (
          <div className="mb-2 text-[10px] uppercase tracking-wide text-gray-500">
            route:{" "}
            <span className="rounded bg-gray-100 px-1 py-0.5">{m.route}</span>
          </div>
        )}
        {/* Markdown rendering */}
        <Md>{rendered}</Md>
        {m.used_docs && m.used_docs.length > 0 && (
          <div className="mt-3 grid gap-2">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">
              출처 (References)
            </div>
            {m.used_docs.map((d, i) => (
              <SourceCard key={i} doc={d} index={i + 1} />
            ))}
          </div>
        )}
      </div>
      {isUser && (
        <div className="mt-1 h-8 w-8 flex-none rounded-full bg-gray-800" />
      )}
    </div>
  );
}

function Composer({
  onSend,
  disabled,
}: {
  onSend: (q: string) => void;
  disabled?: boolean;
}) {
  const [value, setValue] = useState("");
  const taRef = useRef<HTMLTextAreaElement | null>(null);
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    const handler = () => {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 180) + "px";
    };
    handler();
    el.addEventListener("input", handler);
    return () => el.removeEventListener("input", handler);
  }, []);
  const handleKeyDown: React.KeyboardEventHandler<HTMLTextAreaElement> = (
    e
  ) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };
  const submit = () => {
    const q = value.trim();
    if (!q) return;
    onSend(q);
    setValue("");
  };
  return (
    <div className="sticky bottom-0 z-10 border-t bg-white p-3">
      <div className="mx-auto flex max-w-3xl items-end gap-2">
        <textarea
          ref={taRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="질문을 입력하세요… (Shift+Enter로 줄바꿈)"
          rows={1}
          className="min-h-[44px] max-h-[180px] flex-1 resize-none rounded-2xl border px-4 py-3 text-sm outline-none focus:border-indigo-400"
        />
        <button
          onClick={submit}
          disabled={disabled}
          className={cls(
            "rounded-2xl bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm",
            disabled && "opacity-50"
          )}
        >
          Send
        </button>
      </div>
    </div>
  );
}

// ====== Root App ======
export default function App() {
  const { sid, update, clear } = useLocalSessionId();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState(false);
  const [typewriter, setTypewriter] = useState(true);
  const scRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    scRef.current?.scrollTo({
      top: scRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const handleNew = () => {
    clear();
    setMessages([]);
  };

  const ask = async (q: string) => {
    setPending(true);
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: q,
    };
    setMessages((m) => [...m, userMsg]);
    try {
      const { data, returnedSessionId } = await postChat(q, sid);
      if (returnedSessionId && returnedSessionId !== sid)
        update(returnedSessionId);
      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.answer,
        route: data.route,
        used_docs: data.used_docs,
      };
      setMessages((m) => [...m, assistantMsg]);
    } catch (err: any) {
      const assistantErr: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `요청 실패: ${err?.message || err}`,
      };
      setMessages((m) => [...m, assistantErr]);
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="flex h-dvh flex-col">
      <Header onNew={handleNew} />

      <main className="mx-auto grid w-full max-w-6xl flex-1 grid-cols-12 gap-4 px-4 py-4">
        <aside className="col-span-3 hidden flex-col gap-2 md:flex">
          <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">
            세션
          </div>
          <div className="rounded-xl border bg-white p-3 text-sm">
            <div className="mb-1 text-gray-500">Session ID</div>
            <div className="truncate font-mono text-xs">
              {sid || "(새 세션)"}
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <button
                onClick={handleNew}
                className="rounded-lg border px-2 py-1 text-xs hover:bg-gray-50"
              >
                새 대화 시작
              </button>
              <a
                className="rounded-lg border px-2 py-1 text-xs hover:bg-gray-50"
                href={`${API_BASE}/docs`}
                target="_blank"
                rel="noreferrer"
              >
                Swagger
              </a>
              <label className="ml-auto flex items-center gap-2 text-xs text-gray-600">
                <input
                  type="checkbox"
                  checked={typewriter}
                  onChange={(e) => setTypewriter(e.target.checked)}
                />
                typewriter
              </label>
            </div>
          </div>
        </aside>

        <section className="col-span-12 md:col-span-9">
          <div
            ref={scRef}
            className="flex h-[calc(100dvh-200px)] flex-col gap-4 overflow-y-auto rounded-2xl border bg-white p-4"
          >
            {messages.length === 0 ? (
              <div className="m-auto max-w-xl text-center text-gray-600">
                <h2 className="mb-2 text-lg font-semibold">
                  무엇을 도와드릴까요?
                </h2>
                <p className="text-sm">
                  저작권법 판례 기반 RAG 어시스턴트에 질문해 보세요. 예: "업무상
                  과실치사에 대해 알려주고, 실제 사례를 하나 알려줘."
                </p>
              </div>
            ) : (
              messages.map((m) => (
                <MessageBubble key={m.id} m={m} typewriter={typewriter} />
              ))
            )}
            {pending && (
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <div className="h-2 w-2 animate-pulse rounded-full bg-gray-400" />
                답변 생성 중…
              </div>
            )}
          </div>
          <Composer onSend={ask} disabled={pending} />
        </section>
      </main>

      <footer className="border-t py-3 text-center text-xs text-gray-500">
        © {new Date().getFullYear()} Copyright Advisor · Built with React,
        Tailwind, LangChain RAG
      </footer>
    </div>
  );
}
