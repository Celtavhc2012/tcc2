import React from "react";

export const Header = () => {
  return (
    <header class="bg-white">
      <nav
        class="mx-auto flex max-w-7xl items-center justify-between p-6 lg:px-8"
        aria-label="Global"
      >
        <div class="flex lg:flex-1">
          <a href="#" class="-m-1.5 p-1.5">
            <span class="sr-only">Your Company</span>
            <img
              class="h-12 w-auto"
              src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTv5v2nhEOYqsyOSJncHZaNOY69-O4K5Zx6tA&s"
              alt=""
            />
          </a>
        </div>
       
        <div class="hidden lg:flex lg:flex-1 lg:justify-end">
          <a href="#" class="text-sm/6 font-semibold text-gray-900">
            Selecione seu loge xes <span aria-hidden="true">&rarr;</span>
          </a>
        </div>
      </nav>

    
    </header>
  );
};
