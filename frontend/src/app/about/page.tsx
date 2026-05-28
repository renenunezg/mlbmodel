import Image from "next/image";
import Link from "next/link";
import { PostCard } from "@/components/post-card";
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
              <PostCard key={post.slug} post={post} />
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
