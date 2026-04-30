import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import { runEvalForGame } from "@/lib/eval-game";

// Live per-game eval, fired from the games page when a game flips to Final.
// Delegates to runEvalForGame; this handler only enforces origin + parses input.

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

function originAllowed(req: Request): boolean {
  const origin = req.headers.get("origin") ?? req.headers.get("referer") ?? "";
  if (!origin) return false;
  const allow = [
    "http://localhost:3000",
    "https://renenunez.work",
    "https://www.renenunez.work",
  ];
  if (allow.some((a) => origin.startsWith(a))) return true;
  return /https:\/\/[a-z0-9-]+\.vercel\.app/.test(origin);
}

export async function POST(req: Request) {
  if (!originAllowed(req)) {
    return NextResponse.json({ error: "forbidden origin" }, { status: 403 });
  }

  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !serviceKey) {
    return NextResponse.json(
      { error: "supabase service-role env not configured" },
      { status: 500 },
    );
  }

  let body: { game_pk?: number };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad json" }, { status: 400 });
  }
  const game_pk = body.game_pk;
  if (!game_pk || typeof game_pk !== "number") {
    return NextResponse.json({ error: "missing game_pk" }, { status: 400 });
  }

  const sb = createClient(url, serviceKey, { auth: { persistSession: false } });
  const result = await runEvalForGame(sb, game_pk);
  if ("error" in result && result.error) {
    return NextResponse.json({ error: result.error }, { status: 500 });
  }
  return NextResponse.json(result);
}
