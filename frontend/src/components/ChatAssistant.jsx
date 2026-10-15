import React, { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Bot, MessageCircle, RotateCcw, Send, X } from 'lucide-react';

const REPORT_SUGGESTIONS = ['What is this disease?', 'What should I do now?', 'What stage is it?', 'Is it serious?'];
const GENERAL_SUGGESTIONS = ['What is a retina photo?', 'How do I get my retina checked?', 'Which eye diseases can this find?'];

// Minimal, safe Markdown: paragraphs, "-"/"*"/"1." lists, ### headings and **bold**. No HTML is injected.
function inline(text) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith('**') && part.endsWith('**') ? <strong key={i}>{part.slice(2, -2)}</strong> : part,
  );
}

function renderMarkdown(text) {
  const blocks = [];
  let list = null;
  const flush = () => {
    if (list) blocks.push(list);
    list = null;
  };
  text.split('\n').forEach((raw, i) => {
    const line = raw.trim();
    const bullet = line.match(/^([-*•]|\d+[.)])\s+(.*)$/);
    if (bullet) {
      const ordered = /\d/.test(bullet[1][0]);
      if (!list || list.ordered !== ordered) {
        flush();
        list = { ordered, items: [] };
      }
      list.items.push(bullet[2]);
      return;
    }
    flush();
    if (!line) return;
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    blocks.push(heading ? { heading: heading[1], key: i } : { text: line, key: i });
  });
  flush();
  return blocks.map((b, i) => {
    if (b.items) {
      const Tag = b.ordered ? 'ol' : 'ul';
      return (
        <Tag key={i} className={`${b.ordered ? 'list-decimal' : 'list-disc'} space-y-1 pl-5`}>
          {b.items.map((item, j) => <li key={j}>{inline(item)}</li>)}
        </Tag>
      );
    }
    if (b.heading) return <p key={i} className="font-semibold">{inline(b.heading)}</p>;
    return <p key={i}>{inline(b.text)}</p>;
  });
}

function greeting(result) {
  if (!result) {
    return 'Hello! I can answer questions about eye health and retina screening. Upload a retina photo and I can explain the report with you.';
  }
  if (result.status === 'rejected_for_quality') {
    return 'Hello! The photo in this report could not be read clearly. Ask me what that means or how to get a better photo.';
  }
  const name = result.primary_disease === 'normal' ? 'no eye disease found' : result.primary_disease_name;
  return `Hello! I have read this report (${name}). Ask me what it means, what to do next, or anything that worries you. You can write in your own language.`;
}

export default function ChatAssistant({ result }) {
  const screeningId = result?.id ?? null;
  const [open, setOpen] = useState(false);
  // One conversation (and error) per report, so switching reports never mixes contexts.
  const convKey = screeningId ?? 'general';
  const [conversations, setConversations] = useState({});
  const [errors, setErrors] = useState({});
  const messages = conversations[convKey] || [];
  const error = errors[convKey] || null;
  const setMessages = (list) => setConversations((c) => ({ ...c, [convKey]: list }));
  const setError = (msg) => setErrors((e) => ({ ...e, [convKey]: msg }));
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [status, setStatus] = useState(null);
  const listRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!open || status) return;
    fetch('/api/chat/status')
      .then((r) => (r.ok ? r.json() : null))
      .then(setStatus)
      .catch(() => {});
  }, [open, status]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [conversations, convKey, sending, error]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  const send = async (text) => {
    const content = text.trim();
    if (!content || sending) return;
    const next = [...messages, { role: 'user', content }];
    setMessages(next);
    setInput('');
    setError(null);
    setSending(true);
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ screening_id: screeningId, messages: next }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = data?.detail;
        throw new Error(typeof detail === 'object' ? detail.message : detail || `Server error (${res.status})`);
      }
      setMessages([...next, { role: 'assistant', content: data.reply }]);
    } catch (e) {
      setError(e.message || 'The assistant is unavailable right now.');
    } finally {
      setSending(false);
    }
  };

  const suggestions = screeningId && result?.status !== 'rejected_for_quality' ? REPORT_SUGGESTIONS : GENERAL_SUGGESTIONS;

  return (
    <>
      {!open && (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="fixed bottom-5 right-5 z-50 flex items-center gap-2 rounded-full bg-sky-600 px-5 py-3 text-sm font-semibold text-white shadow-xl transition-transform hover:scale-[1.03] hover:bg-sky-500"
          style={{ marginBottom: 'env(safe-area-inset-bottom, 0px)' }}
        >
          <MessageCircle className="h-5 w-5" />
          {screeningId ? 'Ask about this report' : 'Ask RetinaLens AI'}
        </button>
      )}

      {open && (
        <section
          aria-label="RetinaLens assistant"
          className="fixed bottom-5 right-5 z-50 flex h-[min(600px,calc(100vh-6rem))] w-[min(400px,calc(100vw-2.5rem))] flex-col overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 shadow-2xl"
        >
          <header className="theme-purple-island flex items-center justify-between gap-3 bg-slate-950 px-4 py-3 text-white">
            <div className="flex items-center gap-2.5">
              <span className="flex h-8 w-8 items-center justify-center rounded-full bg-sky-600">
                <Bot className="h-4 w-4 text-white" />
              </span>
              <div>
                <p className="text-sm font-semibold leading-tight">RetinaLens Assistant</p>
                <p className="text-[11px] text-slate-400">
                  {screeningId ? `Explaining report #${screeningId.slice(0, 8)}` : 'General eye-health questions'}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              {messages.length > 0 && (
                <button type="button" onClick={() => { setMessages([]); setError(null); }} title="New conversation"
                  className="rounded-md p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white">
                  <RotateCcw className="h-4 w-4" />
                </button>
              )}
              <button type="button" onClick={() => setOpen(false)} title="Close" aria-label="Close assistant"
                className="rounded-md p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white">
                <X className="h-4 w-4" />
              </button>
            </div>
          </header>

          <div ref={listRef} className="flex-1 space-y-3 overflow-y-auto px-4 py-4 text-sm">
            <div className="max-w-[88%] rounded-2xl rounded-tl-sm bg-slate-950/60 px-3.5 py-2.5 leading-relaxed text-slate-200">
              {greeting(result)}
            </div>

            {messages.map((m, i) =>
              m.role === 'user' ? (
                <div key={i} className="ml-auto max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-tr-sm bg-sky-600 px-3.5 py-2.5 text-white">
                  {m.content}
                </div>
              ) : (
                <div key={i} className="max-w-[92%] space-y-2 rounded-2xl rounded-tl-sm bg-slate-950/60 px-3.5 py-2.5 leading-relaxed text-slate-200">
                  {renderMarkdown(m.content)}
                </div>
              ),
            )}

            {sending && (
              <div className="flex w-fit items-center gap-1 rounded-2xl rounded-tl-sm bg-slate-950/60 px-4 py-3" aria-label="Assistant is typing">
                {[0, 150, 300].map((d) => (
                  <span key={d} className="h-2 w-2 animate-bounce rounded-full bg-slate-500" style={{ animationDelay: `${d}ms` }} />
                ))}
              </div>
            )}

            {error && (
              <div className="flex items-start gap-2 rounded-xl border border-amber-500/40 bg-amber-500/10 px-3 py-2.5 text-xs leading-relaxed text-amber-200">
                <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-400" />
                <span>{error}</span>
              </div>
            )}

            {messages.length === 0 && !sending && (
              <div className="flex flex-wrap gap-2 pt-1">
                {suggestions.map((s) => (
                  <button key={s} type="button" onClick={() => send(s)}
                    className="rounded-full border border-slate-700 bg-slate-950/50 px-3 py-1.5 text-xs text-slate-200 transition-colors hover:border-sky-500 hover:text-white">
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>

          <form
            onSubmit={(e) => { e.preventDefault(); send(input); }}
            className="border-t border-slate-800 bg-slate-900 p-3"
          >
            {status && !status.configured && (
              <p className="mb-2 text-[11px] text-amber-300">The assistant is not set up on the server yet.</p>
            )}
            <div className="flex items-end gap-2">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    send(input);
                  }
                }}
                rows={1}
                maxLength={2000}
                placeholder="Type your question…"
                className="max-h-28 min-h-[40px] flex-1 resize-none rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-white placeholder-slate-500 focus:border-sky-500 focus:outline-none"
              />
              <button type="submit" disabled={sending || !input.trim()} aria-label="Send"
                className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-sky-600 text-white transition-colors hover:bg-sky-500 disabled:opacity-40">
                <Send className="h-4 w-4" />
              </button>
            </div>
            <p className="mt-2 text-center text-[10px] text-slate-500">
              AI answers can be wrong. Always confirm with an eye doctor.
            </p>
          </form>
        </section>
      )}
    </>
  );
}
