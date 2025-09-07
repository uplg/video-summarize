import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

interface SummaryResponse {
  title: string;
  summary: string;
  transcript: string;
}

function App() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) {
      setError("Please enter a valid YouTube URL");
      return;
    }

    setLoading(true);
    setError("");
    setSummary(null);

    try {
      const response = await fetch("http://localhost:8000/summarize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) {
        throw new Error("Error processing video");
      }

      const data = await response.json();
      setSummary(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-screen p-4">
      <div className="mx-auto space-y-6">
        <Card className="w-full">
          <CardHeader>
            <CardTitle>Enter YouTube Video URL</CardTitle>
            <CardDescription>
              Paste the URL of the video you want to summarize
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="flex gap-2">
                <Input
                  type="url"
                  placeholder="https://www.youtube.com/watch?v=..."
                  value={url}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setUrl(e.target.value)
                  }
                  className="flex-1"
                  disabled={loading}
                />
                <Button type="submit" disabled={loading || !url.trim()}>
                  {loading ? "Processing..." : "Summarize"}
                </Button>
              </div>
              {error && <div className="text-red-600 text-sm">{error}</div>}
            </form>
          </CardContent>
        </Card>

        {loading && (
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center justify-center space-x-2">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                <span>Extracting and summarizing...</span>
              </div>
            </CardContent>
          </Card>
        )}

        {summary && (
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl">{summary.title}</CardTitle>
                <CardDescription>
                  Automatically generated summary
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="prose max-w-none">
                  <Textarea
                    value={summary.summary}
                    readOnly
                    className="min-h-[300px] resize-none"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Full Transcription</CardTitle>
                <CardDescription>Complete video text</CardDescription>
              </CardHeader>
              <CardContent>
                <Textarea
                  value={summary.transcript}
                  readOnly
                  className="min-h-[200px] resize-none text-sm"
                />
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
