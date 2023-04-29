
% idea: scene objects encoded like:
% object(ID)
% has_attribute(ID, class, car)
% has_attribute(ID, color, red)
% has_attribute(ID, size, large)
% has_attribute(ID, any, full)
% has_relation(ID, in, ID')
% has_relation(ID, in_front_of, ID'')

#defined has_attribute/3.
#defined has_relation/3.
#defined is_attribute_value/2.

% ------- Rules for operations -------
% scene
#defined scene/1.

state(T,ID) :- scene(T), object(ID).

% and
#defined and/3.

bool(T,yes) :- and(T, RT0, RT1), bool(RT0,yes), bool(RT1,yes).
bool(T,no) :- and(T, RT0, RT1), not bool(T,yes).

% or
#defined or/3.

bool(T,yes) :- or(T, RT0, RT1), bool(RT0,yes).
bool(T,yes) :- or(T, RT0, RT1), bool(RT1,yes).
bool(T,no) :- or(T, RT0, RT1), not bool(T,yes).

% choose_attr
#defined choose_attr/5.
1 {has_attribute(ID, ATTR, VALUE); has_attribute(ID, ATTR, VALUE')} 1:- choose_attr(T, RT, ATTR, VALUE, VALUE'), state(RT, ID).
attr_value(T, VALUE) :- choose_attr(T, RT, ATTR, VALUE, VALUE'), state(RT, ID), has_attribute(ID, ATTR, VALUE).
attr_value(T, VALUE') :- choose_attr(T, RT, ATTR, VALUE, VALUE'), state(RT, ID), has_attribute(ID, ATTR, VALUE').

% choose_rel
#defined choose_rel/6.
1 {has_relation(ID', REL, ID): object(ID'); has_relation(ID', REL', ID): object(ID')} 1 :- choose_rel(T, RT, CLASS, REL, REL', subject), state(RT, ID), has_attribute(ID', class, CLASS).
rel(T, REL) :- choose_rel(T, RT, CLASS, REL, REL', subject), state(RT, ID), has_attribute(ID', class, CLASS), has_relation(ID', REL, ID).
rel(T, REL') :- choose_rel(T, RT, CLASS, REL, REL', subject), state(RT, ID), has_attribute(ID', class, CLASS), has_relation(ID', REL', ID).

1 {has_relation(ID, REL, ID'): object(ID'); has_relation(ID, REL', ID'): object(ID')} 1:- choose_rel(T, RT, CLASS, REL, REL', subject), state(RT, ID), has_attribute(ID', class, CLASS).
rel(T, REL) :- choose_rel(T, RT, CLASS, REL, REL', object), state(RT, ID), has_attribute(ID', class, CLASS), has_relation(ID, REL, ID').
rel(T, REL') :- choose_rel(T, RT, CLASS, REL, REL', object), state(RT, ID), has_attribute(ID', class, CLASS), has_relation(ID, REL', ID').

% compare
#defined compare/5.

state(T,ID) :- compare(T, RT0, RT1, VALUE, true), state(RT0, ID), state(RT1, ID'), has_attribute(ID, _, VALUE), not has_attribute(ID', _, VALUE).
state(T,ID') :- compare(T, RT0, RT1, VALUE, true), state(RT0, ID), state(RT1, ID'), not has_attribute(ID, _, VALUE), has_attribute(ID', _, VALUE).

state(T,ID') :- compare(T, RT0, RT1, VALUE, false), state(RT0, ID), state(RT1, ID'), has_attribute(ID, _, VALUE), not has_attribute(ID', _, VALUE).
state(T,ID) :- compare(T, RT0, RT1, VALUE, false), state(RT0, ID), state(RT1, ID'), not has_attribute(ID, _, VALUE), has_attribute(ID', _, VALUE).

% common
#defined common/3.

attr(T, ATTR) :- common(T, RT0, RT1), state(RT0, ID), state(RT1, ID'), has_attribute(ID, ATTR, VALUE), has_attribute(ID', ATTR, VALUE'), VALUE == VALUE'.

% exist
#defined exist/2.

bool(T,yes) :- exist(T, RT), state(RT,ID).
bool(T,no) :- exist(T, RT), not bool(T,yes).

% filter
#defined filter/4.

state(T,ID) :- filter(T, RT, ATTR, VALUE), state(RT, ID), has_attribute(ID, ATTR, VALUE).

#defined filter_any/3.

state(T,ID) :- filter_any(T, RT, VALUE), state(RT, ID), has_attribute(ID, ATTR, VALUE).

% negate
#defined negate/3.
state(T, ID) :- negate(T, RT, SRT), state(SRT, ID), not state(RT, ID).

% query_attr
#defined query_attr/3.

1 { has_attribute(ID, ATTR, VALUE) : is_attribute_value(ATTR, VALUE)} 1 :- query_attr(T, RT, ATTR), state(RT, ID), ATTR != name, ATTR != class, ATTR != hposition, ATTR != vposition.
attr_value(T,VALUE) :- query_attr(T, RT, ATTR), state(RT, ID), has_attribute(ID, ATTR, VALUE).

% relate
#defined relate/5.

state(T, ID') :- relate(T, RT, CLASS, REL, subject), state(RT, ID), has_attribute(ID', class, CLASS), has_relation(ID', REL, ID).


state(T, ID') :- relate(T, RT, CLASS, REL, object), state(RT, ID), has_attribute(ID', class, CLASS), has_relation(ID, REL, ID').

% relate_any
#defined relate_any/4.

state(T, ID') :- relate_any(T, RT, REL, subject), state(RT, ID), has_relation(ID', REL, ID).

state(T, ID') :- relate_any(T, RT, REL, object), state(RT, ID), has_relation(ID, REL, ID').

% relate_attr
#defined relate_attr/4.

state(T, ID') :- relate_attr(T, RT, CLASS, ATTR), state(RT, ID), has_attribute(ID, ATTR, VALUE), has_attribute(ID', class, CLASS), has_attribute(ID', ATTR, VALUE'), VALUE==VALUE', ID!=ID'.


% select
#defined select/3.

state(T,ID) :- select(T, RT, CLASS), state(RT, ID), has_attribute(ID, class, CLASS).

% two_different
#defined two_different/4.

bool(T, yes) :- two_different(T, RT0, RT1, ATTR), state(RT0, ID), state(RT1, ID'), has_attribute(ID, ATTR, VALUE), has_attribute(ID', ATTR, VALUE'), VALUE != VALUE'.
bool(T, yes) :- two_different(T, RT0, RT1, ATTR), state(RT0, ID), state(RT1, ID'), has_attribute(ID, ATTR, _), not has_attribute(ID', ATTR, _).
bool(T, yes) :- two_different(T, RT0, RT1, ATTR), state(RT0, ID), state(RT1, ID'), not has_attribute(ID, ATTR, _), has_attribute(ID', ATTR, _).
bool(T,no) :- two_different(T, RT0, RT1, ATTR), not bool(T,yes).

% two_same
#defined two_same/4.

bool(T, yes) :- two_same(T, RT0, RT1, ATTR), state(RT0, ID), state(RT1, ID'), has_attribute(ID, ATTR, VALUE), has_attribute(ID', ATTR, VALUE'), VALUE == VALUE'.
bool(T,no) :- two_same(T, RT0, RT1, ATTR), not bool(T,yes).

% all_different
#defined all_different/3.

bool(T,no) :- all_different(T, RT, ATTR), state(RT, ID), state(RT, ID'), has_attribute(ID, ATTR, VALUE), has_attribute(ID', ATTR, VALUE).
bool(T,yes) :- all_different(T, RT, ATTR), not bool(T,no).

% all_same
#defined all_same/3.

bool(T,no) :- all_same(T, RT, ATTR), state(RT, ID), state(RT, ID'), has_attribute(ID, ATTR, VALUE), not has_attribute(ID', ATTR, VALUE).
bool(T,yes) :- all_same(T, RT, ATTR), not bool(T,no).

% unique
#defined unique/2.

state(T,ID) :- unique(T, RT), state(RT,ID).
:- unique(T, RT), state(RT,ID), state(RT,ID'), ID!=ID'.

% verify_attr
#defined verify_attr/4.

bool(T, yes) :- verify_attr(T, RT, ATTR, VALUE), state(RT, ID), has_attribute(ID, ATTR, VALUE).
bool(T,no) :- verify_attr(T, RT, ATTR, VALUE), not bool(T,yes).

% verify_rel
#defined verify_rel/5.

bool(T, yes) :- verify_rel(T, RT, CLASS, REL, subject), state(RT, ID), has_attribute(ID', class, CLASS), has_relation(ID', REL, ID).
bool(T,no) :- verify_rel(T, RT, CLASS, REL, subject), not bool(T,yes).

bool(T, yes) :- verify_rel(T, RT, CLASS, REL, object), state(RT, ID), has_attribute(ID', class, CLASS), has_relation(ID, REL, ID').
bool(T,no) :- verify_rel(T, RT, CLASS, REL, object), not bool(T,yes).


% Derive answer (T must equal the last point in time)
#defined end/1.

ans(V) :- end(T), attr_value(T,V).
ans(V) :- end(T), attr(T,V).
ans(V) :- end(T), rel(T,V).
ans(V) :- end(T), bool(T,V).

:- not ans(_).

#show ans/1.
% #show state/2.