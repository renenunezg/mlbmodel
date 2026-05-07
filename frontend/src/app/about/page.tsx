import Image from "next/image";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { posts } from "./posts";

const contact = [
  { label: "Email", value: "renenunezgalaviz@gmail.com", href: "mailto:renenunezgalaviz@gmail.com" },
  { label: "GitHub", value: "github.com/renenunezg", href: "https://github.com/renenunezg" },
  { label: "LinkedIn", value: "linkedin.com/in/renenunezg", href: "https://linkedin.com/in/renenunezg" },
  { label: "Twitter", value: "@nunezanalytics", href: "https://twitter.com/nunezanalytics" },
];

export default function AboutPage() {
  return (
    <main className="mx-auto w-full max-w-3xl min-w-0 px-4 py-8">
      <div className="mb-10 flex flex-col gap-6 sm:flex-row sm:items-start sm:gap-8">
        {/* Photo */}
        <div className="shrink-0">
          <Image
            src="/portrait.jpeg"
            alt="Rene Nunez"
            width={96}
            height={96}
            className="rounded-full object-cover"
            priority
          />
        </div>

        {/* Name + bio + contact */}
        <div className="flex-1 min-w-0">
          <h1 className="font-heading text-2xl tracking-tight">Rene Nunez</h1>
          <p className="mt-2 text-sm text-muted-foreground leading-relaxed max-w-xl">
            {/* Write your one-liner here */}
          </p>

          <div className="mt-5">
            <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground mb-2">
              Contact
            </p>
            <div className="flex flex-col gap-1.5">
              {contact.map(({ label, value, href }) => (
                <div key={label} className="flex items-center gap-3 text-sm">
                  <span className="font-mono text-xs uppercase tracking-wider text-muted-foreground w-16 shrink-0">
                    {label}
                  </span>
                  <Link
                    href={href}
                    target={href.startsWith("mailto") ? undefined : "_blank"}
                    rel="noopener noreferrer"
                    className="text-foreground hover:underline underline-offset-4"
                  >
                    {value}
                  </Link>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <section>
        <h2 className="font-heading text-base tracking-tight mb-4">Research</h2>
        {posts.length === 0 ? (
          <p className="text-sm text-muted-foreground">Nothing published yet.</p>
        ) : (
          <div className="flex flex-col gap-4">
            {posts.map((post) => (
              <Card key={post.slug}>
                <CardHeader className="border-b pb-3">
                  <p className="font-mono text-xs text-muted-foreground">{post.date}</p>
                  <CardTitle className="text-base mt-0.5">{post.title}</CardTitle>
                  <CardDescription>{post.summary}</CardDescription>
                </CardHeader>
                <CardContent className="pt-4 text-sm text-muted-foreground leading-relaxed">
                  <div className="space-y-3 [&_p]:leading-relaxed [&_h2]:font-heading [&_h2]:text-base [&_h2]:tracking-tight [&_h2]:text-foreground [&_h2]:mt-6 [&_h2]:mb-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:space-y-1 [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:space-y-1 [&_strong]:text-foreground [&_table]:w-full [&_table]:text-xs [&_table]:my-3 [&_th]:border [&_th]:border-border [&_th]:px-2 [&_th]:py-1 [&_th]:text-left [&_th]:font-mono [&_th]:font-normal [&_th]:text-foreground [&_td]:border [&_td]:border-border [&_td]:px-2 [&_td]:py-1 [&_td]:font-mono [&_img]:my-4 [&_img]:rounded-md [&_img]:border [&_img]:border-border [&_img]:max-w-full">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {post.body}
                    </ReactMarkdown>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
