"use client";
import useUser from "@/hooks/useUser";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { v4 as uuid } from "uuid";
import {
  SignedIn,
  SignedOut,
  UserButton,
  SignInButton,
  SignUpButton,
  useAuth,
  useClerk
} from "@clerk/nextjs";
import { motion } from "framer-motion"; // Import Framer Motion

export default function Home() {
  const { fullName, setFullName } = useUser();
  const [roomID, setRoomID] = useState("");
  const router = useRouter();
  const { isSignedIn } = useAuth();
  const { openSignIn } = useClerk();

  useEffect(() => {
    setFullName("");
  }, []);

  const handleJoinClick = () => {
    if (!isSignedIn) {
      openSignIn();
      return;
    }

    if (roomID) {
      router.push(`/room/${roomID}`);
    } else {
      alert("Please enter a valid Room ID.");
    }
  };

  const handleCreateMeeting = () => {
    if (!isSignedIn) {
      openSignIn();
      return;
    }

    const newRoomID = uuid();
    router.push(`/room/${newRoomID}`);
  };

  return (
    <div className="w-full h-screen">
      <nav className="fixed top-0 left-0 w-full bg-opacity-50 backdrop-blur-md bg-gray-900/50 shadow-md z-50 py-4">
        <div className="max-w-screen-xl mx-auto px-6 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Image src="/logo.svg" alt="SignSync Logo" width={40} height={40} />
            <span className="text-xl font-bold text-white">SignSync</span>
          </div>
          <div className="flex items-center gap-4">
            <SignedOut>
              <SignInButton mode="modal">
                <button className="px-4 py-2 text-white bg-blue-400 rounded-md hover:bg-blue-700">
                  Sign In
                </button>
              </SignInButton>
              <SignUpButton mode="modal">
                <button className="px-4 py-2 text-white bg-green-600 rounded-md hover:bg-green-700">
                  Sign Up
                </button>
              </SignUpButton>
            </SignedOut>
            <SignedIn>
              <UserButton />
            </SignedIn>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="bg-gray-950 text-white">
        <div className="mx-auto max-w-screen-xl px-4 py-32 flex-col gap-24 flex h-screen items-center">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="mx-auto mt-25 max-w-4xl text-center"
          >
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.8 }}
              className="bg-gradient-to-r from-green-300 via-blue-500 to-purple-600 bg-clip-text font-extrabold text-transparent text-5xl"
            >
              Have a smooth meeting
            </motion.h1>
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.8 }}
              className="bg-gradient-to-r from-green-300 via-blue-500 to-purple-600 bg-clip-text font-extrabold text-transparent text-5xl"
            >
              <span className="block">with team members</span>
            </motion.h1>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 0.8 }}
              className="mx-auto mt-6 max-w-fit sm:text-xl/relaxed text-center"
            >
              Welcome to SignSync – the AI-powered real-time sign language
              interpreter for virtual meetings. Our platform seamlessly converts
              sign language gestures into instant captions and speech, ensuring
              clear and inclusive communication for both deaf and hearing
              participants. Powered by advanced AI, real-time hand tracking,
              SignSync makes every virtual meeting accessible and effortless.
            </motion.p>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8, duration: 0.8 }}
              className="flex items-center justify-center gap-4 mt-6"
            >
              <input
                type="text"
                id="name"
                onChange={(e) => setFullName(e.target.value.toString())}
                className="border bg-white rounded-md focus:border-transparent focus:outline-none focus:ring-0 px-4 py-2 w-full text-black"
                placeholder="Enter your name"
              />
            </motion.div>

            {fullName && fullName.length >= 3 && (
              <>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 1, duration: 0.8 }}
                  className="flex items-center justify-center gap-4 mt-6"
                >
                  <input
                    type="text"
                    id="roomid"
                    value={roomID}
                    onChange={(e) => setRoomID(e.target.value)}
                    className="bg-white border rounded-md focus:border-transparent focus:outline-none focus:ring-0 px-4 py-2 w-full text-black"
                    placeholder="Enter room ID to join a meeting"
                  />
                  <button
                    className="rounded-md bg-blue-600 px-10 py-[11px] text-sm font-medium text-white focus:outline-none sm:w-auto"
                    onClick={handleJoinClick}
                    disabled={!roomID}
                  >
                    Join
                  </button>
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 1.2, duration: 0.8 }}
                  className="mt-4 flex items-center justify-center"
                >
                  <button
                    className="text-lg font-medium hover:text-blue-400 hover:underline"
                    onClick={handleCreateMeeting}
                  >
                    Or create a new meeting
                  </button>
                </motion.div>
              </>
            )}
          </motion.div>
        </div>
      </section>
    </div>
  );
}
