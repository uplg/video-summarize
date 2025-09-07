import { useState, useEffect } from "react";
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
import { Progress } from "@/components/ui/progress";

interface TaskResponse {
  task_id: string;
  status: string;
  message: string;
}

interface TaskStatusResponse {
  task_id: string;
  status: string;
  progress: number;
  message: string;
  result?: {
    title: string;
    summary: string;
    transcript: string;
  };
  error?: string;
  created_at: string;
  updated_at: string;
}

interface SummaryResponse {
  title: string;
  summary: string;
  transcript: string;
}

function App() {
  const [url, setUrl] = useState("");
  const [currentTask, setCurrentTask] = useState<TaskStatusResponse | null>(null);
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [error, setError] = useState("");
  const [isPolling, setIsPolling] = useState(false);

  // Poll task status
  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    
    if (isPolling && currentTask && currentTask.status !== "completed" && currentTask.status !== "failed") {
      intervalId = setInterval(async () => {
        try {
          const response = await fetch(`http://localhost:8000/task/${currentTask.task_id}`);
          if (response.ok) {
            const taskStatus: TaskStatusResponse = await response.json();
            setCurrentTask(taskStatus);
            
            if (taskStatus.status === "completed" && taskStatus.result) {
              setSummary(taskStatus.result);
              setIsPolling(false);
            } else if (taskStatus.status === "failed") {
              setError(taskStatus.error || "Task failed");
              setIsPolling(false);
            }
          }
        } catch (err) {
          console.error("Error polling task status:", err);
        }
      }, 2000); // Poll every 2 seconds
    }
    
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isPolling, currentTask]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) {
      setError("Please enter a valid YouTube URL");
      return;
    }

    setError("");
    setSummary(null);
    setCurrentTask(null);

    try {
      const response = await fetch("http://localhost:8000/summarize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) {
        throw new Error("Error starting task");
      }

      const taskResponse: TaskResponse = await response.json();
      
      // Get initial task status
      const statusResponse = await fetch(`http://localhost:8000/task/${taskResponse.task_id}`);
      if (statusResponse.ok) {
        const taskStatus: TaskStatusResponse = await statusResponse.json();
        setCurrentTask(taskStatus);
        setIsPolling(true);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    }
  };

  const getStatusMessage = (status: string) => {
    switch (status) {
      case "pending":
        return "Task queued for processing...";
      case "extracting_audio":
        return "Extracting audio from YouTube video...";
      case "transcribing":
        return "Transcribing audio to text...";
      case "generating_summary":
        return "Generating blog post summary...";
      case "completed":
        return "Summary generated successfully!";
      case "failed":
        return "Task failed";
      default:
        return "Processing...";
    }
  };

  const isProcessing = Boolean(currentTask && currentTask.status !== "completed" && currentTask.status !== "failed");

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
                  disabled={isProcessing}
                />
                <Button type="submit" disabled={isProcessing || !url.trim()}>
                  {isProcessing ? "Processing..." : "Summarize"}
                </Button>
              </div>
              {error && <div className="text-red-600 text-sm">{error}</div>}
            </form>
          </CardContent>
        </Card>

        {isProcessing && currentTask && (
          <Card>
            <CardHeader>
              <CardTitle>Processing Video</CardTitle>
              <CardDescription>Task ID: {currentTask.task_id}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>{getStatusMessage(currentTask.status)}</span>
                  <span>{currentTask.progress}%</span>
                </div>
                <Progress value={currentTask.progress} className="w-full" />
              </div>
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                <span>{currentTask.message}</span>
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
