export type Post = {
  slug: string;
  date: string;       // "YYYY-MM-DD"
  title: string;
  summary: string;
  body: string;       // plain text or light markdown - rendered as paragraphs split on \n\n
};

// Add new entries at the top. Each entry appears as a card on the About page.
export const posts: Post[] = [
  // {
  //   slug: "example-post",
  //   date: "2026-04-21",
  //   title: "Your post title",
  //   summary: "One sentence shown in the card preview.",
  //   body: "First paragraph.\n\nSecond paragraph.",
  // },
];
