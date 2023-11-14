#!/bin/sh
#
# ---------------------------------------------------------------------
# PyCharm startup script.
# ---------------------------------------------------------------------
#

message()
{
  TITLE="Cannot start PyCharm"
  if [ -t 1 ]; then
    echo "ERROR: $TITLE\n$1"
  elif [ -n `which zenity` ]; then
    zenity --error --title="$TITLE" --text="$1"
  elif [ -n `which kdialog` ]; then
    kdialog --error --title "$TITLE" "$1"
  elif [ -n `which xmessage` ]; then
    xmessage -center "ERROR: $TITLE: $1"
  elif [ -n `which notify-send` ]; then
    notify-send "ERROR: $TITLE: $1"
  else
    echo "ERROR: $TITLE\n$1"
  fi
}

UNAME=`which uname`
GREP=`which egrep`
GREP_OPTIONS=""
CUT=`which cut`
READLINK=`which readlink`
XARGS=`which xargs`
DIRNAME=`which dirname`
MKTEMP=`which mktemp`
RM=`which rm`
CAT=`which cat`
TR=`which tr`

if [ -z "$UNAME" -o -z "$GREP" -o -z "$CUT" -o -z "$MKTEMP" -o -z "$RM" -o -z "$CAT" -o -z "$TR" ]; then
  message "Required tools are missing - check beginning of \"$0\" file for details."
  exit 1
fi

OS_TYPE=`"$UNAME" -s`

if [ -e "/usr/lib/jvm/java-7-oracle/jre/bin/java" ]; then
	PYCHARM_JDK="/usr/lib/jvm/java-7-oracle/jre/"
fi
# ---------------------------------------------------------------------
# Locate a JDK installation directory which will be used to run the IDE.
# Try (in order): PYCHARM_JDK, JDK_HOME, JAVA_HOME, "java" in PATH.
# ---------------------------------------------------------------------
if [ -n "$PYCHARM_JDK" -a -x "$PYCHARM_JDK/bin/java" ]; then
  JDK="$PYCHARM_JDK"
elif [ -n "$JDK_HOME" -a -x "$JDK_HOME/bin/java" ]; then
  JDK="$JDK_HOME"
elif [ -n "$JAVA_HOME" -a -x "$JAVA_HOME/bin/java" ]; then
  JDK="$JAVA_HOME"
else
  JAVA_BIN_PATH=`which java`
  if [ -n "$JAVA_BIN_PATH" ]; then
    if [ "$OS_TYPE" = "FreeBSD" -o "$OS_TYPE" = "MidnightBSD" ]; then
      JAVA_LOCATION=`JAVAVM_DRYRUN=yes java | "$GREP" '^JAVA_HOME' | "$CUT" -c11-`
      if [ -x "$JAVA_LOCATION/bin/java" ]; then
        JDK="$JAVA_LOCATION"
      fi
    elif [ "$OS_TYPE" = "SunOS" ]; then
      JAVA_LOCATION="/usr/jdk/latest"
      if [ -x "$JAVA_LOCATION/bin/java" ]; then
        JDK="$JAVA_LOCATION"
      fi
    elif [ "$OS_TYPE" = "Darwin" ]; then
      JAVA_LOCATION=`/usr/libexec/java_home`
      if [ -x "$JAVA_LOCATION/bin/java" ]; then
        JDK="$JAVA_LOCATION"
      fi
    fi

    if [ -z "$JDK" -a -x "$READLINK" -a -x "$XARGS" -a -x "$DIRNAME" ]; then
      JAVA_LOCATION=`"$READLINK" -f "$JAVA_BIN_PATH"`
      case "$JAVA_LOCATION" in
        */jre/bin/java)
          JAVA_LOCATION=`echo "$JAVA_LOCATION" | "$XARGS" "$DIRNAME" | "$XARGS" "$DIRNAME" | "$XARGS" "$DIRNAME"`
          if [ ! -d "$JAVA_LOCATION/bin" ]; then
            JAVA_LOCATION="$JAVA_LOCATION/jre"
          fi
          ;;
        *)
          JAVA_LOCATION=`echo "$JAVA_LOCATION" | "$XARGS" "$DIRNAME" | "$XARGS" "$DIRNAME"`
          ;;
      esac
      if [ -x "$JAVA_LOCATION/bin/java" ]; then
        JDK="$JAVA_LOCATION"
      fi
    fi
  fi
fi

if [ -z "$JDK" ]; then
  message "No JDK found. Please validate either PYCHARM_JDK, JDK_HOME or JAVA_HOME environment variable points to valid JDK installation."
  exit 1
fi

VERSION_LOG=`"$MKTEMP" -t java.version.log.XXXXXX`
"$JDK/bin/java" -version 2> "$VERSION_LOG"
"$GREP" "64-Bit|x86_64" "$VERSION_LOG" > /dev/null
BITS=$?
"$RM" -f "$VERSION_LOG"
if [ $BITS -eq 0 ]; then
  BITS="64"
else
  BITS=""
fi

# ---------------------------------------------------------------------
# Ensure IDE_HOME points to the directory where the IDE is installed.
# ---------------------------------------------------------------------
SCRIPT_LOCATION=$0
if [ -x "$READLINK" ]; then
  while [ -L "$SCRIPT_LOCATION" ]; do
    SCRIPT_LOCATION=`"$READLINK" -e "$SCRIPT_LOCATION"`
  done
fi

IDE_HOME=`dirname "$SCRIPT_LOCATION"`/..
IDE_BIN_HOME=`dirname "$SCRIPT_LOCATION"`

# ---------------------------------------------------------------------
# Collect JVM options and properties.
# ---------------------------------------------------------------------
if [ -n "$PYCHARM_PROPERTIES" ]; then
  IDE_PROPERTIES_PROPERTY="-Didea.properties.file=\"$PYCHARM_PROPERTIES\""
fi

MAIN_CLASS_NAME="$PYCHARM_MAIN_CLASS_NAME"
if [ -z "$MAIN_CLASS_NAME" ]; then
  MAIN_CLASS_NAME="com.intellij.idea.Main"
fi

VM_OPTIONS_FILE="$PYCHARM_VM_OPTIONS"
if [ -z "$VM_OPTIONS_FILE" ]; then
  VM_OPTIONS_FILE="$IDE_BIN_HOME/pycharm$BITS.vmoptions"
fi

if [ -r "$VM_OPTIONS_FILE" ]; then
  VM_OPTIONS=`"$CAT" "$VM_OPTIONS_FILE" | "$GREP" -v "^#.*" | "$TR" '\n' ' '`
  VM_OPTIONS="$VM_OPTIONS -Djb.vmOptionsFile=\"$VM_OPTIONS_FILE\""
fi

IS_EAP="false"
if [ "$IS_EAP" = "true" ]; then
  OS_NAME=`echo $OS_TYPE | "$TR" '[:upper:]' '[:lower:]'`
  AGENT_LIB="yjpagent-$OS_NAME$BITS"
  if [ -r "$IDE_BIN_HOME/lib$AGENT_LIB.so" ]; then
    AGENT="-agentlib:$AGENT_LIB=disablej2ee,disablealloc,delay=10000,sessionname=PyCharm30"
  fi
fi

COMMON_JVM_ARGS="\"-Xbootclasspath/a:$IDE_HOME/lib/boot.jar\" -Didea.paths.selector=PyCharm30 $IDE_PROPERTIES_PROPERTY"
IDE_JVM_ARGS="-Didea.platform.prefix=PyCharmCore -Didea.no.jre.check=true"
ALL_JVM_ARGS="$VM_OPTIONS $COMMON_JVM_ARGS $IDE_JVM_ARGS $AGENT $REQUIRED_JVM_ARGS"

CLASSPATH="$IDE_HOME/lib/bootstrap.jar"
CLASSPATH="$CLASSPATH:$IDE_HOME/lib/extensions.jar"
CLASSPATH="$CLASSPATH:$IDE_HOME/lib/util.jar"
CLASSPATH="$CLASSPATH:$IDE_HOME/lib/jdom.jar"
CLASSPATH="$CLASSPATH:$IDE_HOME/lib/log4j.jar"
CLASSPATH="$CLASSPATH:$IDE_HOME/lib/trove4j.jar"
CLASSPATH="$CLASSPATH:$IDE_HOME/lib/jna.jar"
if [ -n "$PYCHARM_CLASSPATH" ]; then
  CLASSPATH="$CLASSPATH:$PYCHARM_CLASSPATH"
fi
export CLASSPATH

LD_LIBRARY_PATH="$IDE_BIN_HOME:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH

# ---------------------------------------------------------------------
# Run the IDE.
# ---------------------------------------------------------------------
while true ; do
  eval "$JDK/bin/java" $ALL_JVM_ARGS -Djb.restart.code=88 $MAIN_CLASS_NAME "$@"
  test $? -ne 88 && break
done
