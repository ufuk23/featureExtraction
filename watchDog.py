import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class OnMyWatch:
    # Set the directory on watch
    watchDirectory = "/mnt/muesfs/mues/temp/"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.watchDirectory, recursive = True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
                print("watching...")
        except:
            self.observer.stop()
            print("Observer Stopped")

        self.observer.join()

    #function to return files in a directory
    def fileInDirectory(my_dir: str):
        onlyfiles = [f for f in listdir(my_dir) if isfile(join(my_dir, f))]
        return(onlyfiles)


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):

        print(event)

        if event.is_directory:
            return None

        elif event.event_type == 'created':
            # Event is created, you can process it now
            print("Watchdog received created event - % s." % event.src_path)
        elif event.event_type == 'modified':
            # Event is modified, you can process it now
            print("Watchdog received modified event - % s." % event.src_path)

if __name__ == '__main__':
    watch = OnMyWatch()
    watch.run()
