"use client";
import useUser from "@/hooks/useUser";
import { useRouter } from "next/navigation";
import { SignedIn, SignedOut, useAuth, useClerk } from "@clerk/nextjs";
import { ZegoUIKitPrebuilt } from "@zegocloud/zego-uikit-prebuilt";
import React, { use, useEffect, useRef, useState } from "react";
import { v4 as uuid } from "uuid";

const Page = ({ params }: { params: Promise<{ roomid: string }> }) => {
  const { isSignedIn, userId } = useAuth();
  const { fullName } = useUser();
  const { roomid } = use(params);
  const router = useRouter();
  const {openSignIn} = useClerk();

  const zpRef = useRef<ReturnType<typeof ZegoUIKitPrebuilt.create> | null>(
    null
  );
  const roomContainerRef = useRef<HTMLDivElement | null>(null);

  // 🔥 Captions State
  const [captions, setCaptions] = useState<string>("");
  const [showCaptions, setShowCaptions] = useState<boolean>(true);
  const [inMeeting, setInMeeting] = useState(false); // Track if user is in the meeting

  useEffect(() => {
    if (!isSignedIn) {
      openSignIn();
    }
  }, [isSignedIn, router]);

  useEffect(() => {
    if (!roomContainerRef.current || zpRef.current) return;

    const appID = parseInt(process.env.NEXT_PUBLIC_ZEGO_APP_ID!);
    const serverSecret = process.env.NEXT_PUBLIC_ZEGO_SERVER_SECERET!;

    const kitToken = ZegoUIKitPrebuilt.generateKitTokenForTest(
      appID,
      serverSecret,
      roomid,
      userId || uuid(),
      fullName || `user${Date.now()}`,
      720
    );

    const zp = ZegoUIKitPrebuilt.create(kitToken);
    zpRef.current = zp;

    zp.joinRoom({
      container: roomContainerRef.current,
      sharedLinks: [
        {
          name: "Sharable Link",
          url: `${window.location.origin}${window.location.pathname}?roomID=${roomid}`,
        },
        {
          name: "Meeting ID",
          url: roomid,
        }
      ],
      scenario: {
        mode: ZegoUIKitPrebuilt.VideoConference,
      },
      // ✅ Only show captions AFTER joining the meeting
      onJoinRoom: () => setInMeeting(true),
      onLeaveRoom: () => setInMeeting(false),
    });

    // 🔥 Simulate Captions (Replace with AI Later)
    const interval = setInterval(() => {
      const sampleCaptions = [
        "Hello, everyone!",
        "Testing real-time captions...",
        "Sign language captions will be displayed here.",
      ];
      setCaptions(
        sampleCaptions[Math.floor(Math.random() * sampleCaptions.length)]
      );
    }, 5000);

    return () => {
      if (zpRef.current) {
        zpRef.current.destroy();
        zpRef.current = null;
      }
      clearInterval(interval);
    };
  }, [roomid, fullName, isSignedIn]);

  return (
    <div className="relative w-full h-screen">
      <SignedIn>
        {/* Meeting UI */}
        <div
          ref={roomContainerRef}
          className="w-full h-full absolute inset-0"
        />

        {/* 🔥 Show caption toggle ONLY inside the meeting */}
        {inMeeting && (
          <>
            <button
              onClick={() => setShowCaptions(!showCaptions)}
              className="z-5 absolute bottom-23 left-8 bg-auto text-white px-4 py-2 rounded-md shadow-lg hover:bg-gray-400/10 transition"
            >
              {showCaptions ? "Hide Captions" : "Show Captions"}
            </button>

            {/* 🔥 Captions Overlay */}
            {showCaptions && (
              <div className=" absolute bottom-25 left-1/2 transform -translate-x-1/2 bg-blend-overlay bg-opacity-90 text-white text-lg px-6 py-3 rounded-lg shadow-lg max-w-[80%] text-center z-5">
                {captions}
              </div>
            )}
          </>
        )}
      </SignedIn>
    </div>
  );
};

export default Page;
