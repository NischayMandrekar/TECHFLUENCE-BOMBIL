"use client";
import useUser from "@/hooks/useUser";
import { ZegoUIKitPrebuilt } from "@zegocloud/zego-uikit-prebuilt";
import React, { useEffect, useRef, use } from "react";
import { v4 as uuid } from "uuid";

const Page = ({ params }: { params: Promise<{ roomid: string }> }) => {
  const { fullName } = useUser();
  const { roomid } = use(params);

  // 🔥 Store Zego instance
  const zpRef = useRef<ReturnType<typeof ZegoUIKitPrebuilt.create> | null>(null);
  const roomContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!roomContainerRef.current || zpRef.current) return; // ✅ Prevent duplicate `joinRoom()`

    const appID = parseInt(process.env.NEXT_PUBLIC_ZEGO_APP_ID!);
    const serverSecret = process.env.NEXT_PUBLIC_ZEGO_SERVER_SECERET!;

    const kitToken = ZegoUIKitPrebuilt.generateKitTokenForTest(
      appID,
      serverSecret,
      roomid,
      uuid(), // Replace with authenticated user ID later
      fullName || `user${Date.now()}`,
      720
    );

    const zp = ZegoUIKitPrebuilt.create(kitToken);
    zpRef.current = zp; // ✅ Store instance

    zp.joinRoom({
      container: roomContainerRef.current,
      sharedLinks: [
        {
          name: "Sharable Link",
          url: `${window.location.origin}${window.location.pathname}?roomID=${roomid}`,
        },
      ],
      scenario: {
        mode: ZegoUIKitPrebuilt.VideoConference,
      },
    });

    // ✅ Cleanup function
    return () => {
      if (zpRef.current) {
        zpRef.current.destroy(); // ✅ Properly cleanup instance
        zpRef.current = null; // Reset instance
      }
    };
  }, [roomid, fullName]); // ✅ Only runs once when `roomid` changes

  return <div className="w-full h-screen" ref={roomContainerRef}></div>;
};

export default Page;
