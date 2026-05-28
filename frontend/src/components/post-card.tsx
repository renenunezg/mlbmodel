"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

type Post = {
  slug: string;
  date: string;
  title: string;
  summary: string;
  body: string;
};

const PROSE_CLASSES =
  "space-y-3 [&_p]:leading-relaxed [&_h2]:font-heading [&_h2]:text-base [&_h2]:tracking-tight [&_h2]:text-foreground [&_h2]:mt-6 [&_h2]:mb-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:space-y-1 [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:space-y-1 [&_strong]:text-foreground [&_table]:w-full [&_table]:text-xs [&_table]:my-3 [&_th]:border [&_th]:border-border [&_th]:px-2 [&_th]:py-1 [&_th]:text-left [&_th]:font-mono [&_th]:font-normal [&_th]:text-foreground [&_td]:border [&_td]:border-border [&_td]:px-2 [&_td]:py-1 [&_td]:font-mono [&_img]:my-4 [&_img]:rounded-md [&_img]:border [&_img]:border-border [&_img]:max-w-full";

export function PostCard({ post }: { post: Post }) {
  const [expanded, setExpanded] = useState(false);

  const firstParagraph = post.body.split(/\n{2,}/)[0];
  const hasMore = firstParagraph.length < post.body.trim().length;
  const visibleBody = expanded ? post.body : firstParagraph;

  return (
    <Card>
      <CardHeader className="border-b pb-3">
        <p className="font-mono text-xs text-muted-foreground">{post.date}</p>
        <CardTitle className="text-base mt-0.5">{post.title}</CardTitle>
        <CardDescription>{post.summary}</CardDescription>
      </CardHeader>
      <CardContent className="pt-4 text-sm text-muted-foreground leading-relaxed">
        <div className={PROSE_CLASSES}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{visibleBody}</ReactMarkdown>
        </div>
        {hasMore && (
          <button
            onClick={() => setExpanded((v) => !v)}
            className="mt-4 font-mono text-xs uppercase tracking-wider text-muted-foreground underline underline-offset-4 hover:text-foreground transition-colors"
          >
            {expanded ? "Show less" : "Read more"}
          </button>
        )}
      </CardContent>
    </Card>
  );
}
